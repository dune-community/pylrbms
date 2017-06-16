#!/usr/bin/env python


import numpy as np

from dune.xt.la import IstlDenseVectorDouble as Vector
from dune.xt.grid import make_boundary_info_on_dd_subdomain_layer as make_subdomain_boundary_info
from dune.gdt import (
    RS2017_make_elliptic_swipdg_matrix_operator_on_neighborhood as make_elliptic_swipdg_matrix_operator_on_neighborhood,
    RS2017_make_elliptic_swipdg_vector_functional_on_neighborhood
        as make_elliptic_swipdg_vector_functional_on_neighborhood,
    RS2017_make_l2_vector_functional_on_neighborhood as make_l2_vector_functional_on_neighborhood,
    RS2017_make_neighborhood_system_assembler as make_neighborhood_system_assembler,
    make_discrete_function,
    project,
)

from pymor.bindings.dunext import DuneXTMatrixOperator
from pymor.core.interfaces import BasicInterface
from pymor.operators.constructions import LincombOperator, VectorFunctional
from pymor.reductors.system import GenericRBSystemReductor


def doerfler_marking(indicators, theta):
	assert 0.0 < theta <= 1.0
	indices = list(range(len(indicators)))
	indicators = [ii**2 for ii in indicators]
	indicators, indices = [list(x) for x in zip(*sorted(zip(indicators, indices),
                                                            key=lambda pair: pair[0],
                                                            reverse=True))]
	total = np.sum(indicators)
	sums = np.array([np.sum(indicators[:ii+1]) for ii in np.arange(len(indicators))])
	where = sums > theta*total
	if np.any(where):
		return indices[:np.argmax(where)+1]
	else:
		return indices


def solve_for_local_correction(grid, subdomain, local_boundary_info, affine_lambda, kappa, f, d, block_space, reductor, reduced_U, mu):
    neighborhood = grid.neighborhood_of(subdomain)
    neighborhood_space = block_space.restricted_to_neighborhood(neighborhood)
    # Compute current solution restricted to the neighborhood to be usable as Dirichlet values for the correction
    # problem.
    current_solution = []
    for nn in neighborhood:
        local_space_id = d.solution_space.subspaces[nn].id
        current_solution.append(reductor.reconstruct_local(reduced_U, local_space_id))
    current_solution = [v._list for v in current_solution]
    assert np.all(len(v) == 1 for v in current_solution)
    current_solution = [v[0].impl for v in current_solution]
    current_solution = neighborhood_space.project_onto_neighborhood(current_solution, neighborhood)
    current_solution = make_discrete_function(neighborhood_space, current_solution)
    # Solve the local corrector problem.
    #   LHS
    ops = []
    for lambda_ in affine_lambda['functions']:
        ops.append(make_elliptic_swipdg_matrix_operator_on_neighborhood(
            grid, subdomain, local_boundary_info,
            neighborhood_space,
            lambda_, kappa,
            over_integrate=0))
    ops_coeffs = affine_lambda['coefficients'].copy()
    #   RHS
    funcs = []
    for lambda_ in affine_lambda['functions']:
        funcs.append(make_elliptic_swipdg_vector_functional_on_neighborhood(
            grid, subdomain, local_boundary_info,
            neighborhood_space,
            current_solution, lambda_, kappa,
            over_integrate=0))
    funcs_coeffs = affine_lambda['coefficients'].copy()
    funcs.append(make_l2_vector_functional_on_neighborhood(
        grid, subdomain,
        neighborhood_space,
        f,
        over_integrate=0))
    funcs_coeffs.append(1.)
    #   assemble in one grid walk
    neighborhood_assembler = make_neighborhood_system_assembler(grid, subdomain, neighborhood_space)
    for op in ops:
        neighborhood_assembler.append(op)
    for func in funcs:
        neighborhood_assembler.append(func)
    neighborhood_assembler.assemble()
    # solve
    local_space_id = d.solution_space.subspaces[subdomain].id
    lhs = LincombOperator([DuneXTMatrixOperator(o.matrix(), source_id=local_space_id, range_id=local_space_id) for o in ops], ops_coeffs)
    rhs = LincombOperator([VectorFunctional(lhs.range.make_array([v.vector()])) for v in funcs], funcs_coeffs)
    correction = lhs.apply_inverse(rhs.as_source_array(mu), mu=mu)
    assert len(correction) == 1
    # restrict to subdomain
    local_sizes = [block_space.local_space(nn).size() for nn in neighborhood]
    local_starts = [int(np.sum(local_sizes[:nn])) for nn in range(len(local_sizes))]
    local_starts.append(neighborhood_space.mapper.size)
    localized_corrections_as_np = np.array(correction._list[0].impl, copy=False)
    localized_corrections_as_np = [localized_corrections_as_np[local_starts[nn]:local_starts[nn+1]] for nn in range(len(local_sizes))]
    subdomain_index_in_neighborhood = np.where(np.array(list(neighborhood)) == subdomain)[0]
    assert len(subdomain_index_in_neighborhood) == 1
    subdomain_index_in_neighborhood = subdomain_index_in_neighborhood[0]
    subdomain_correction = Vector(local_sizes[subdomain_index_in_neighborhood], 0.)
    subdomain_correction_as_np = np.array(subdomain_correction, copy=False)
    subdomain_correction_as_np[:] = localized_corrections_as_np[subdomain_index_in_neighborhood][:]
    return d.solution_space.subspaces[subdomain].make_array([subdomain_correction])


class AdaptiveEnrichment(BasicInterface):

    def __init__(self, grid_and_problem_data, discretization, block_space, reductor, rd,
            target_error, marking_doerfler_theta, marking_max_age, fake_estimator=None):
        self.grid_and_problem_data = grid_and_problem_data
        self.discretization = discretization
        self.block_space = block_space
        self.local_boundary_info = make_subdomain_boundary_info(self.grid_and_problem_data['grid'],
                                                                {'type': 'xt.grid.boundaryinfo.alldirichlet'})
        self.reductor = reductor
        self.rd = rd
        self.target_error = target_error
        self.marking_doerfler_theta = marking_doerfler_theta
        self.marking_max_age = marking_max_age
        self.fake_estimator = fake_estimator

    def _enrich_once(self, U, mu, indicators, age_count):
        marked_subdomains = set(doerfler_marking(indicators, self.marking_doerfler_theta))
        num_dorfler_marked = len(marked_subdomains)
        self.logger.info3('marked {}/{} subdomains due to Dörfler marking'.format(num_dorfler_marked,
            self.block_space.num_blocks))
        for ii in np.where(age_count > self.marking_max_age)[0]:
            marked_subdomains.add(ii)
        num_age_marked = len(marked_subdomains) - num_dorfler_marked
        self.logger.info3('   and {}/{} additionally due to age marking'.format(num_age_marked, self.block_space.num_blocks - num_dorfler_marked))
        new_reductor = GenericRBSystemReductor(self.discretization, {id_: b.copy() for id_, b in self.reductor.bases.items()})
        self.logger.info3('solving local corrector problems on {} subdomain{} ...'.format(
            len(marked_subdomains), 's' if len(marked_subdomains) > 1 else ''))
        for ii in marked_subdomains:
            local_correction = solve_for_local_correction(
                    self.grid_and_problem_data['grid'], ii, self.local_boundary_info,
                    self.grid_and_problem_data['lambda'], self.grid_and_problem_data['kappa'], self.grid_and_problem_data['f'],
                    self.discretization, self.block_space, self.reductor, U, mu)
            new_reductor.extend_basis_local(local_correction)
        self.reductor = new_reductor
        if self.fake_estimator:
            estimator = self.fake_estimator
            estimator.reductor = self.reductor
        else:
            estimator = self.discretization.estimator
        self.rd = self.reductor.reduce()
        self.rd = self.rd.with_(estimator=estimator)
        # clear age count
        for ii in range(self.block_space.num_blocks):
            if ii in marked_subdomains:
                age_count[ii] = 1
            else:
                age_count[ii] += 1
        return len(marked_subdomains)

    def estimate(self, U, mu, decompose=False):
        return self.rd.estimate(U, mu=mu, decompose=decompose)

    def solve(self, mu, enrichment_steps=np.inf, callback=None):
        mu = self.discretization.parse_parameter(mu)
        enrichment_step = 1
        age_count = np.ones(self.block_space.num_blocks)
        local_problem_solves = 0
        rb_size = self.rd.solution_space.dim
        with self.logger.block('solving {}-dimensional system for mu = {} ...'.format(rb_size, mu)) as _:
            while True:
                U = self.rd.solve(mu)
                eta, _, indicators = self.estimate(U, mu=mu, decompose=True)
                if callback:
                    callback(self.rd, U, mu, {'eta': eta,
                                              'local_problem_solves': local_problem_solves,
                                              'global RB size': self.rd.solution_space.dim,
                                              'local RB sizes': [len(rb) for rb in self.reductor.bases]})
                if eta <= self.target_error:
                    self.logger.info3('estimated error {} below target error of {}, no enrichment required ...'.format(eta, self.target_error))
                    return U, self.rd, self.reductor
                if enrichment_step > enrichment_steps:
                    self.logger.warn('estimated error {} above target error of {}, but stopping since enrichment_steps={} reached!'.format(
                        eta, self.target_error, enrichment_steps))
                    return U, self.rd, self.reductor
                enrichment_step += 1
                self.logger.info3('estimated error {} above target error of {}, enriching ...'.format(eta, self.target_error))
                local_problem_solves = self._enrich_once(U, mu, indicators, age_count)
                self.logger.info3('added {} local basis functions, system size increase: {} --> {}'.format(
                    self.rd.solution_space.dim - rb_size, rb_size, self.rd.solution_space.dim))
                rb_size = self.rd.solution_space.dim


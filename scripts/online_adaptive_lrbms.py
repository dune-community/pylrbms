#!/usr/bin/env python

import numpy as np
np.seterr(all='raise')

from dune.gdt import (
    RS2017_make_elliptic_swipdg_matrix_operator_on_neighborhood as make_elliptic_swipdg_matrix_operator_on_neighborhood,
    RS2017_make_elliptic_swipdg_vector_functional_on_neighborhood
        as make_elliptic_swipdg_vector_functional_on_neighborhood,
    RS2017_make_l2_vector_functional_on_neighborhood as make_l2_vector_functional_on_neighborhood,
    RS2017_make_neighborhood_system_assembler as make_neighborhood_system_assembler,
    project,
)

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.bindings.dunext import DuneXTVector, DuneXTMatrixOperator
from pymor.core.exceptions import ExtensionError
from pymor.core.logger import getLogger
from pymor.operators.constructions import LincombOperator, VectorFunctional
from pymor.reductors.system import GenericRBSystemReductor
from pymor.vectorarrays.list import ListVectorArray

from OS2015_academic_problem import (
        make_expression_function_1x1,
        init_grid_and_problem,
)
from discretize_elliptic import (
        Vector,
        discretize,
        make_discrete_function,
)


config = {'num_coarse_grid_elements': [2, 2],
          'num_grid_refinements': 2,
          'num_grid_subdomains': [2, 2],
          'num_grid_oversampling_layers': 2, # num_grid_oversampling_layers has to exactly cover one subdomain!
          'initial_RB_order': 0,
          'enrichment_target_error': 2., # ([4, 4], 2, [2, 2]): 0.815510144764 | ([4, 4], 6, [8, 8]): 2.25996532203
          'marking_doerfler_theta': 0.33,
          'marking_max_age': 0}


grid_and_problem_data = init_grid_and_problem(config)
grid = grid_and_problem_data['grid']
# grid.visualize('grid')

d, block_space, local_boundary_info = discretize(grid_and_problem_data)

print('estimating some errors:')
errors = []
for mu in (grid_and_problem_data['mu_min'], grid_and_problem_data['mu_max']):
    mu = d.parse_parameter(mu)
    print('  {}: '.format(mu), end='', flush=True)
    U = d.solve(mu)
    estimate = d.estimate(U, mu=mu)
    print(estimate)
    errors.append(estimate)

print('')

U = d.solution_space.empty()
if config['initial_RB_order'] >= 0:
    print('initializing local reduced bases with DG shape functions of up to order {} ... '.format(config['initial_RB_order']), end='')

    reductor = GenericRBSystemReductor(d, products=[d.operators['local_energy_dg_product_{}'.format(ii)]
                                                    for ii in range(block_space.num_blocks)])
    for ii in range(block_space.num_blocks):
        local_space = block_space.local_space(ii)
        # order 0 basis
        reductor.extend_basis_local(ListVectorArray([DuneXTVector(Vector(local_space.size(), 1.)), ],
                                                    U._blocks[ii].space))
        if config['initial_RB_order'] >= 1:
            # order 1 basis
            tmp_discrete_function = make_discrete_function(local_space)
            for expression in ('x[0]', 'x[1]', 'x[0]*x[1]'):
                func = make_expression_function_1x1(grid, 'x', expression, order=2)
                project(func, tmp_discrete_function)
                reductor.extend_basis_local(ListVectorArray([DuneXTVector(tmp_discrete_function.vector_copy()), ],
                                                            U._blocks[ii].space))
            del tmp_discrete_function

print('done')

# print('adding some global solution snapshots to reduced basis ...')
# for mu in (grid_and_problem_data['mu_min'], grid_and_problem_data['mu_max']):
#     U = d.solve(mu)
#     try:
#         reductor.extend_basis(U)
#     except ExtensionError:
#         pass
print('')

print('reducing:')
rd = reductor.reduce()
rd = rd.with_(estimator=d.estimator)
print('')

print('estimating some reduced errors:')
mus = [grid_and_problem_data['mu_min'], grid_and_problem_data['mu_max']]
reduced_errors = []
for mu in mus:
    mu = d.parse_parameter(mu)
    print('  {}: '.format(mu), end='', flush=True)
    U = rd.solve(mu)
    estimate = rd.estimate(U, mu=mu)
    print(estimate)
    reduced_errors.append(estimate)
print('')


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
        local_space_id = d.solution_space.subspaces[ii].id
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
            grid, ii, local_boundary_info,
            neighborhood_space,
            lambda_, kappa,
            over_integrate=0))
    ops_coeffs = affine_lambda['coefficients'].copy()
    #   RHS
    funcs = []
    for lambda_ in affine_lambda['functions']:
        funcs.append(make_elliptic_swipdg_vector_functional_on_neighborhood(
            grid, ii, local_boundary_info,
            neighborhood_space,
            current_solution, lambda_, kappa,
            over_integrate=0))
    funcs_coeffs = affine_lambda['coefficients'].copy()
    funcs.append(make_l2_vector_functional_on_neighborhood(
        grid, ii,
        neighborhood_space,
        f,
        over_integrate=0))
    funcs_coeffs.append(1.)
    #   assemble in one grid walk
    neighborhood_assembler = make_neighborhood_system_assembler(grid, ii, neighborhood_space)
    for op in ops:
        neighborhood_assembler.append(op)
    for func in funcs:
        neighborhood_assembler.append(func)
    neighborhood_assembler.assemble()
    # solve
    local_space_id = d.solution_space.subspaces[ii].id
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
    subdomain_correction = Vector(local_space.size(), 0.)
    subdomain_correction_as_np = np.array(subdomain_correction, copy=False)
    subdomain_correction_as_np[:] = localized_corrections_as_np[subdomain_index_in_neighborhood][:]
    return d.solution_space.subspaces[ii].make_array([subdomain_correction])


print('online phase:')
for mu in rd.parameter_space.sample_randomly(20):
    print('  mu = {}'.format(mu))
    age_count = np.ones(block_space.num_blocks)
    while True:
        U = rd.solve(mu)
        eta, _, indicators = rd.estimate(U, mu=mu, decompose=True)
        print('    estimated error {} '.format(eta), end='')
        if eta < config['enrichment_target_error']:
            print('below tolerance, continuing ...')
            break
        print('too large, enriching ...')
        marked_subdomains = set(doerfler_marking(indicators, config['marking_doerfler_theta']))
        num_dorfler_marked = len(marked_subdomains)
        print('    marked {}/{} subdomains due to Dörfler marking '.format(num_dorfler_marked, block_space.num_blocks), end='')
        for ii in np.where(age_count > config['marking_max_age'])[0]:
            marked_subdomains.add(ii)
        num_age_marked = len(marked_subdomains) - num_dorfler_marked
        print('and {}/{} additionally due to age marking'.format(num_age_marked, block_space.num_blocks - num_dorfler_marked))
        new_reductor = GenericRBSystemReductor(d, {id_: b.copy() for id_, b in reductor.bases.items()})
        for ii in marked_subdomains:
            local_correction = solve_for_local_correction(
                    grid, ii, local_boundary_info,
                    grid_and_problem_data['lambda'], grid_and_problem_data['kappa'], grid_and_problem_data['f'],
                    d, block_space, reductor, U, mu)
            new_reductor.extend_basis_local(local_correction)
        reductor = new_reductor
        rd = reductor.reduce()
        rd = rd.with_(estimator=d.estimator)
        # clear age count
        for ii in range(block_space.num_blocks):
            if ii in marked_subdomains:
                age_count[ii] = 1
            else:
                age_count[ii] += 1
        print('    new reduced system is of size {}x{}'.format(rd.solution_space.dim, rd.solution_space.dim))


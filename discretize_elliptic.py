#!/usr/bin/env python

from itertools import product

import numpy as np

from dune.xt.grid import (
    make_apply_on_dirichlet_intersections_dd_subdomain_boundary_part as make_apply_on_dirichlet_intersections,
    make_boundary_info_on_dd_subdomain_layer as make_subdomain_boundary_info,
    make_walker_on_dd_subdomain_part as make_subdomain_walker
)
from dune.xt.functions import make_expression_function_1x1
from dune.xt.la import (
    IstlDenseVectorDouble as Vector,
    IstlRowMajorSparseMatrixDouble as Matrix,
    SparsityPatternDefault,
)
from dune.gdt import (
    RS2017_apply_diffusive_flux_reconstruction_in_neighborhood as apply_diffusive_flux_reconstruction_in_neighborhood,
    RS2017_make_residual_part_vector_functional_on_subdomain as make_residual_part_vector_functional_on_subdomain,
    RS2017_make_Hdiv_semi_product_matrix_operator_on_subdomain as make_Hdiv_semi_product_matrix_operator_on_subdomain,
    RS2017_make_diffusive_flux_aa_product_matrix_operator_on_subdomain as make_diffusive_flux_aa_product,
    RS2017_make_diffusive_flux_ab_product_matrix_operator_on_subdomain as make_diffusive_flux_ab_product,
    RS2017_make_diffusive_flux_bb_product_matrix_operator_on_subdomain as make_diffusive_flux_bb_product,
    RS2017_apply_l2_product as apply_l2_product,
    RS2017_make_elliptic_matrix_operator_on_subdomain as make_local_elliptic_matrix_operator,
    RS2017_make_elliptic_swipdg_matrix_operator_on_neighborhood as make_elliptic_swipdg_matrix_operator_on_neighborhood,
    RS2017_make_elliptic_swipdg_vector_functional_on_neighborhood as make_elliptic_swipdg_vector_functional_on_neighborhood,
    RS2017_make_l2_vector_functional_on_neighborhood as make_l2_vector_functional_on_neighborhood,
    RS2017_make_neighborhood_system_assembler as make_neighborhood_system_assembler,
    RS2017_make_penalty_product_matrix_operator_on_subdomain as make_penalty_product_matrix_operator,
    RS2017_residual_indicator_min_diffusion_eigenvalue as min_diffusion_eigenvalue,
    RS2017_residual_indicator_subdomain_diameter as subdomain_diameter,
    RS2017_make_divergence_matrix_operator_on_subdomain as make_divergence_matrix_operator_on_subdomain,
    apply_oswald_interpolation_operator,
    make_block_dg_dd_subdomain_part_to_1x1_fem_p1_space as make_block_space,
    make_discrete_function,
    make_elliptic_matrix_operator_istl_row_major_sparse_matrix_double as make_elliptic_matrix_operator,
    make_elliptic_swipdg_affine_factor_matrix_operator as make_elliptic_swipdg_matrix_operator,
    make_l2_matrix_operator,
    make_l2_volume_vector_functional,
    make_local_elliptic_swipdg_affine_factor_boundary_integral_operator_1x1_p1_dg_fem_space_dd_subdomain_coupling_intersection as make_local_elliptic_swipdg_boundary_operator,  # NOQA
    make_local_elliptic_swipdg_affine_factor_inner_integral_operator_1x1_p1_dg_fem_space_dd_subdomain_coupling_intersection as make_local_elliptic_swipdg_coupling_operator,  # NOQA
    make_rt_leaf_view_to_2x1_pdelab_p0_space as make_rt_space,
    make_system_assembler,
    project as dune_project
)

from pymor.basic import *
from pymor.bindings.dunegdt import DuneGDTVisualizer
from pymor.bindings.dunext import DuneXTMatrixOperator, DuneXTVectorSpace
from pymor.core.exceptions import ExtensionError
from pymor.core.interfaces import ImmutableInterface
from pymor.core.logger import getLogger
from pymor.operators.basic import OperatorBase
from pymor.operators.block import BlockOperator, BlockDiagonalOperator, BlockProjectionOperator, BlockRowOperator, BlockColumnOperator
from pymor.parameters.functionals import ProductParameterFunctional
from pymor.reductors.system import GenericRBSystemReductor
from pymor.vectorarrays.block import BlockVectorSpace


class OswaldInterpolationErrorOperator(OperatorBase):

    linear = True

    def __init__(self, subdomain, solution_space, grid, block_space):
        self.subdomain, self.grid, self.block_space = subdomain, grid, block_space
        self.neighborhood = grid.neighborhood_of(subdomain)
        self.source = solution_space.subspaces[subdomain]
        self.range = BlockVectorSpace([solution_space.subspaces[ii] for ii in self.neighborhood],
                                      'OI_{}'.format(subdomain))

    def apply(self, U, mu=None):
        assert U in self.source
        results = self.range.empty(reserve=len(U))
        for u_i in range(len(U)):
            result = self.range.zeros()
            result._blocks[self.neighborhood.index(self.subdomain)].axpy(1, U[u_i])

            for i_ii, ii in enumerate(self.neighborhood):
                ii_neighborhood = self.grid.neighborhood_of(ii)
                ii_neighborhood_space = self.block_space.restricted_to_neighborhood(ii_neighborhood)

                subdomain_uh_with_neighborhood_support = make_discrete_function(
                    ii_neighborhood_space,
                    ii_neighborhood_space.project_onto_neighborhood(
                        [U._list[u_i].impl if nn == self.subdomain else Vector(self.block_space.local_space(nn).size(), 0.)
                         for nn in ii_neighborhood],
                        ii_neighborhood
                    )
                )

                interpolated_u_vector = ii_neighborhood_space.project_onto_neighborhood(
                    [Vector(self.block_space.local_space(nn).size(), 0.) for nn in ii_neighborhood], ii_neighborhood)
                interpolated_u = make_discrete_function(ii_neighborhood_space, interpolated_u_vector)

                apply_oswald_interpolation_operator(
                    self.grid, ii,
                    make_subdomain_boundary_info(self.grid, {'type': 'xt.grid.boundaryinfo.alldirichlet'}),
                    subdomain_uh_with_neighborhood_support,
                    interpolated_u
                )

                local_sizes = np.array([ii_neighborhood_space.local_space(nn).size() for nn in ii_neighborhood])
                offsets = np.hstack(([0], np.cumsum(local_sizes)))
                ind = ii_neighborhood.index(ii)
                result._blocks[i_ii]._list[0].data[:] -= np.frombuffer(interpolated_u_vector)[offsets[ind]:offsets[ind+1]]
            results.append(result)

        return results


class FluxReconstructionOperator(OperatorBase):

    linear = True

    def __init__(self, subdomain, solution_space, grid, block_space, global_rt_space, subdomain_rt_spaces,
                 lambda_xi, kappa):
        self.grid = grid
        self.block_space = block_space
        self.global_rt_space = global_rt_space
        self.subdomain_rt_spaces = subdomain_rt_spaces
        self.subdomain = subdomain
        self.neighborhood = grid.neighborhood_of(subdomain)
        self.lambda_xi = lambda_xi
        self.kappa = kappa

        self.source = solution_space.subspaces[subdomain]
        vector_type = solution_space.subspaces[0].vector_type
        self.range = BlockVectorSpace([DuneXTVectorSpace(vector_type, subdomain_rt_spaces[ii].size(), 'LOCALRT_' + str(ii))
                                       for ii in self.grid.neighborhood_of(subdomain)],
                                      'RT_{}'.format(subdomain))

    def apply(self, U, mu=None):
        assert U in self.source
        result = self.range.empty(reserve=len(U))
        for u_i in range(len(U)):
            subdomain_uhs_with_global_support = \
                make_discrete_function(
                    self.block_space,
                    self.block_space.project_onto_neighborhood(
                        [U._list[u_i].impl if nn == self.subdomain else Vector(self.block_space.local_space(nn).size(), 0.)
                         for nn in range(self.grid.num_subdomains)],
                        [nn for nn in range(self.grid.num_subdomains)]
                    )
                )

            reconstructed_uh_kk_with_global_support = make_discrete_function(self.global_rt_space)
            apply_diffusive_flux_reconstruction_in_neighborhood(
                self.grid, self.subdomain,
                self.lambda_xi, self.kappa,
                subdomain_uhs_with_global_support,
                reconstructed_uh_kk_with_global_support)

            blocks = [s.make_array([self.subdomain_rt_spaces[ii].restrict(reconstructed_uh_kk_with_global_support.vector_copy())])
                      for s, ii in zip(self.range.subspaces, self.grid.neighborhood_of(self.subdomain))]
            result.append(self.range.make_array(blocks))

        return result


class Estimator(ImmutableInterface):


    def __init__(self, min_diffusion_evs, subdomain_diameters, local_eta_rf_squared, lambda_coeffs, mu_bar, mu_hat,
                 flux_reconstruction, oswald_interpolation_error):
        self.min_diffusion_evs = min_diffusion_evs
        self.subdomain_diameters = subdomain_diameters
        self.local_eta_rf_squared = local_eta_rf_squared
        self.lambda_coeffs = lambda_coeffs
        self.mu_bar = mu_bar
        self.mu_hat = mu_hat
        self.flux_reconstruction = flux_reconstruction
        self.oswald_interpolation_error = oswald_interpolation_error
        self.num_subdomains = len(subdomain_diameters)

    def estimate(self, U, mu, discretization, decompose=False):
        d = discretization

        alpha_mu_mu_bar = self.alpha(self.lambda_coeffs, mu, self.mu_bar)
        gamma_mu_mu_bar = self.gamma(self.lambda_coeffs, mu, self.mu_bar)
        alpha_mu_mu_hat = self.alpha(self.lambda_coeffs, mu, self.mu_hat)

        local_eta_nc = np.zeros(self.num_subdomains)
        local_eta_r = np.zeros(self.num_subdomains)
        local_eta_df = np.zeros(self.num_subdomains)

        U_r = self.flux_reconstruction.apply(U, mu=mu)
        U_o = self.oswald_interpolation_error.apply(U)

        for ii in range(self.num_subdomains):
            local_eta_nc[ii] = d.operators['nc_{}'.format(ii)].apply2(U_o, U_o, mu=mu)
            local_eta_r[ii] += self.local_eta_rf_squared[ii]
            local_eta_r[ii] -= 2*d.operators['r1_{}'.format(ii)].apply(U_r, mu=mu).data
            local_eta_r[ii] += d.operators['r2_{}'.format(ii)].apply2(U_r, U_r, mu=mu)
            local_eta_df[ii] += d.operators['df_aa_{}'.format(ii)].apply2(U, U, mu=mu)
            local_eta_df[ii] += d.operators['df_bb_{}'.format(ii)].apply2(U_r, U_r, mu=mu)
            local_eta_df[ii] += 2*d.operators['df_ab_{}'.format(ii)].apply2(U, U_r, mu=mu)

            # eta r, scale
            poincaree_constant = 1./(np.pi**2)
            min_diffusion_ev = self.min_diffusion_evs[ii]
            subdomain_h = self.subdomain_diameters[ii]
            local_eta_r[ii] *= (poincaree_constant/min_diffusion_ev) * subdomain_h**2

        local_eta_nc = np.sqrt(local_eta_nc)
        local_eta_r = np.sqrt(local_eta_r)
        local_eta_df = np.sqrt(local_eta_df)

        eta = 0.
        eta +=     np.sqrt(gamma_mu_mu_bar)  * np.linalg.norm(local_eta_nc)
        eta += (1./np.sqrt(alpha_mu_mu_hat)) * np.linalg.norm(local_eta_r + local_eta_df)
        eta *=  1./np.sqrt(alpha_mu_mu_bar)

        if decompose:
            local_indicators = np.zeros(self.num_subdomains)
            for ii in range(self.num_subdomains):
                local_indicators[ii] = (2./alpha_mu_mu_bar) * (
                               gamma_mu_mu_bar * local_eta_nc[ii]**2
                        + (1./alpha_mu_mu_hat) * (local_eta_r[ii] + local_eta_df[ii])**2)

            return eta, (local_eta_nc, local_eta_r, local_eta_df), local_indicators
        else:
            return eta

    def alpha(self, thetas, mu, mu_bar):
        result = np.inf
        for theta in thetas:
            theta_mu = theta.evaluate(mu)
            theta_mu_bar = theta.evaluate(mu_bar)
            assert theta_mu/theta_mu_bar > 0
            result = np.min((result, theta_mu/theta_mu_bar))
            return result

    def gamma(self, thetas, mu, mu_bar):
        result = -np.inf
        for theta in thetas:
            theta_mu = theta.evaluate(mu)
            theta_mu_bar = theta.evaluate(mu_bar)
            assert theta_mu/theta_mu_bar > 0
            result = np.max((result, theta_mu/theta_mu_bar))
        return result


class DuneDiscretizationBase:

    def visualize(self, U, *args, **kwargs):
        self.visualizer.visualize(self.unblock(U), self, *args, **kwargs)

    def unblock(self, U):
        return self.global_operator.source.from_data(U.data)

    def shape_functions(self, subdomain, order=0):
        assert 0 <= order <= 1
        local_space = self.solution_space.subspaces[subdomain]
        U = local_space.make_array([Vector(local_space.dim, 1.)])

        if order == 1:
            dune_local_space = self.visualizer.space.local_space(subdomain)
            tmp_discrete_function = make_discrete_function(dune_local_space)
            for expression in ('x[0]', 'x[1]', 'x[0]*x[1]'):
                func = make_expression_function_1x1(grid, 'x', expression, order=2)
                dune_project(func, tmp_discrete_function)
                U.append(local_space.make_array([tmp_discrete_function.vector_copy()]))

        return U


class DuneDiscretization(DuneDiscretizationBase, StationaryDiscretization):

    def __init__(self, global_operator, global_rhs,
                 neighborhoods,
                 enrichment_data,  # = grid, local_boundary_info, affine_lambda, kappa, f, block_space
                 operator, rhs,
                 products=None, operators=None,
                 parameter_space=None, estimator=None, visualizer=None, cache_region=None, name=None):
        super().__init__(operator, rhs, products=products, operators=operators,
                         parameter_space=parameter_space, estimator=estimator, visualizer=visualizer,
                         cache_region=cache_region, name=name)
        self.global_operator, self.global_rhs, self.neighborhoods, self.enrichment_data = \
            global_operator, global_rhs, neighborhoods, enrichment_data

    def _solve(self, mu):
        if not self.logging_disabled:
            self.logger.info('Solving {} for {} ...'.format(self.name, mu))

        return self.solution_space.from_data(
            self.global_operator.apply_inverse(self.global_rhs.as_vector(mu=mu), mu=mu).data
        )


    def solve_for_local_correction(self, subdomain, Us, mu=None):
        grid, local_boundary_info, affine_lambda, kappa, f, block_space = self.enrichment_data
        neighborhood = self.neighborhoods[subdomain]
        neighborhood_space = block_space.restricted_to_neighborhood(neighborhood)
        # Compute current solution restricted to the neighborhood to be usable as Dirichlet values for the correction
        # problem.
        current_solution = [U._list for U in Us]
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
        local_space_id = self.solution_space.subspaces[subdomain].id
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
        return self.solution_space.subspaces[subdomain].make_array([subdomain_correction])


def discretize(grid_and_problem_data):
    logger = getLogger('discretize_elliptic.discretize_block_SWIPDG')
    logger.info('discretizing ... ')

    grid, boundary_info, inner_boundary_id = (grid_and_problem_data['grid'],
                                              grid_and_problem_data['boundary_info'],
                                              grid_and_problem_data['inner_boundary_id'])
    local_all_dirichlet_boundary_info = make_subdomain_boundary_info(grid, {'type': 'xt.grid.boundaryinfo.alldirichlet'})
    local_all_neumann_boundary_info = make_subdomain_boundary_info(grid, {'type': 'xt.grid.boundaryinfo.allneumann'})
    neighborhood_boundary_info = make_subdomain_boundary_info(
        grid,
        {'type': 'xt.grid.boundaryinfo.boundarysegmentindexbased',
         'default': 'dirichlet',
         'neumann': '[{} {}]'.format(inner_boundary_id, inner_boundary_id+1)})

    affine_lambda, kappa, f = (grid_and_problem_data['lambda'],
                               grid_and_problem_data['kappa'],
                               grid_and_problem_data['f'])
    lambda_bar, lambda_hat = grid_and_problem_data['lambda_bar'], grid_and_problem_data['lambda_hat']
    mu_bar, mu_hat, parameter_range  = (grid_and_problem_data['mu_bar'],
                                        grid_and_problem_data['mu_hat'],
                                        grid_and_problem_data['parameter_range'])

    block_space = make_block_space(grid)

    local_patterns = [block_space.local_space(ii).compute_pattern('face_and_volume')
                      for ii in range(block_space.num_blocks)]
    coupling_patterns_in_in = {}
    coupling_patterns_out_out = {}
    coupling_patterns_in_out = {}
    coupling_patterns_out_in = {}
    for ii in range(grid.num_subdomains):
        for jj in grid.neighboring_subdomains(ii):
            if ii < jj:  # Assemble primally (visit each coupling only once).
                coupling_patterns_in_in[(ii, jj)] = block_space.local_space(ii).compute_pattern('face_and_volume')
                coupling_patterns_out_out[(ii, jj)] = block_space.local_space(jj).compute_pattern('face_and_volume')
                coupling_patterns_in_out[(ii, jj)] = block_space.compute_coupling_pattern(ii, jj, 'face')
                coupling_patterns_out_in[(ii, jj)] = block_space.compute_coupling_pattern(jj, ii, 'face')
    boundary_patterns = {}
    for ii in grid.boundary_subdomains():
        boundary_patterns[ii] = block_space.local_space(ii).compute_pattern('face_and_volume')

    def discretize_lhs_for_lambda(lambda_):
        local_matrices = [None]*grid.num_subdomains
        local_vectors = [None]*grid.num_subdomains
        boundary_matrices = {}
        coupling_matrices_in_in = {}
        coupling_matrices_out_out = {}
        coupling_matrices_in_out = {}
        coupling_matrices_out_in = {}
        for ii in range(grid.num_subdomains):
            local_matrices[ii] = Matrix(block_space.local_space(ii).size(),
                                        block_space.local_space(ii).size(),
                                        local_patterns[ii])
            local_vectors[ii] = Vector(block_space.local_space(ii).size())
            if ii in grid.boundary_subdomains():
                boundary_matrices[ii] = Matrix(block_space.local_space(ii).size(),
                                               block_space.local_space(ii).size(),
                                               boundary_patterns[ii])
            for jj in grid.neighboring_subdomains(ii):
                if ii < jj:  # Assemble primally (visit each coupling only once).
                    coupling_matrices_in_in[(ii, jj)] = Matrix(block_space.local_space(ii).size(),
                                                               block_space.local_space(ii).size(),
                                                               coupling_patterns_in_in[(ii, jj)])
                    coupling_matrices_out_out[(ii, jj)] = Matrix(block_space.local_space(jj).size(),
                                                                 block_space.local_space(jj).size(),
                                                                 coupling_patterns_out_out[(ii, jj)])
                    coupling_matrices_in_out[(ii, jj)] = Matrix(block_space.local_space(ii).size(),
                                                                block_space.local_space(jj).size(),
                                                                coupling_patterns_in_out[(ii, jj)])
                    coupling_matrices_out_in[(ii, jj)] = Matrix(block_space.local_space(jj).size(),
                                                                block_space.local_space(ii).size(),
                                                                coupling_patterns_out_in[(ii, jj)])

        def assemble_local_contributions(subdomain):
            ipdg_operator = make_elliptic_swipdg_matrix_operator(lambda_, kappa, local_all_neumann_boundary_info,
                                                                 local_matrices[subdomain],
                                                                 block_space.local_space(subdomain))
            l2_functional = make_l2_volume_vector_functional(f, local_vectors[subdomain],
                                                             block_space.local_space(subdomain))
            local_assembler = make_system_assembler(block_space.local_space(subdomain))
            local_assembler.append(ipdg_operator)
            local_assembler.append(l2_functional)
            local_assembler.assemble()

        for ii in range(grid.num_subdomains):
            assemble_local_contributions(ii)

        local_ipdg_coupling_operator = make_local_elliptic_swipdg_coupling_operator(lambda_, kappa)

        def assemble_coupling_contributions(subdomain, neighboring_subdomain):
            coupling_assembler = block_space.coupling_assembler(subdomain, neighboring_subdomain)
            coupling_assembler.append(local_ipdg_coupling_operator,
                                      coupling_matrices_in_in[(subdomain, neighboring_subdomain)],
                                      coupling_matrices_out_out[(subdomain, neighboring_subdomain)],
                                      coupling_matrices_in_out[(subdomain, neighboring_subdomain)],
                                      coupling_matrices_out_in[(subdomain, neighboring_subdomain)])
            coupling_assembler.assemble()

        for ii in range(grid.num_subdomains):
            for jj in grid.neighboring_subdomains(ii):
                if ii < jj:  # Assemble primally (visit each coupling only once).
                    assemble_coupling_contributions(ii, jj)

        local_ipdg_boundary_operator = make_local_elliptic_swipdg_boundary_operator(lambda_, kappa)
        apply_on_dirichlet_intersections = make_apply_on_dirichlet_intersections(boundary_info)

        def assemble_boundary_contributions(subdomain):
            boundary_assembler = block_space.boundary_assembler(subdomain)
            boundary_assembler.append(local_ipdg_boundary_operator,
                                      boundary_matrices[subdomain],
                                      apply_on_dirichlet_intersections)
            boundary_assembler.assemble()

        for ii in grid.boundary_subdomains():
            assemble_boundary_contributions(ii)

        global_pattern = SparsityPatternDefault(block_space.mapper.size)
        for ii in range(grid.num_subdomains):
            block_space.mapper.copy_local_to_global(local_patterns[ii], ii, global_pattern)
            if ii in grid.boundary_subdomains():
                block_space.mapper.copy_local_to_global(boundary_patterns[ii], ii, global_pattern)
            for jj in grid.neighboring_subdomains(ii):
                if ii < jj:  # Assemble primally (visit each coupling only once).
                    block_space.mapper.copy_local_to_global(coupling_patterns_in_in[(ii, jj)], ii, ii, global_pattern)
                    block_space.mapper.copy_local_to_global(coupling_patterns_out_out[(ii, jj)], jj, jj, global_pattern)
                    block_space.mapper.copy_local_to_global(coupling_patterns_in_out[(ii, jj)], ii, jj, global_pattern)
                    block_space.mapper.copy_local_to_global(coupling_patterns_out_in[(ii, jj)], jj, ii, global_pattern)

        system_matrix = Matrix(block_space.mapper.size, block_space.mapper.size, global_pattern)
        rhs_vector = Vector(block_space.mapper.size, 0.)
        for ii in range(grid.num_subdomains):
            block_space.mapper.copy_local_to_global(local_matrices[ii], local_patterns[ii], ii, system_matrix)
            block_space.mapper.copy_local_to_global(local_vectors[ii], ii, rhs_vector)
            if ii in grid.boundary_subdomains():
                block_space.mapper.copy_local_to_global(boundary_matrices[ii], boundary_patterns[ii], ii, ii, system_matrix)
            for jj in grid.neighboring_subdomains(ii):
                if ii < jj:  # Assemble primally (visit each coupling only once).
                    block_space.mapper.copy_local_to_global(coupling_matrices_in_in[(ii, jj)],
                                                            coupling_patterns_in_in[(ii, jj)],
                                                            ii, ii, system_matrix)
                    block_space.mapper.copy_local_to_global(coupling_matrices_out_out[(ii, jj)],
                                                            coupling_patterns_out_out[(ii, jj)],
                                                            jj, jj, system_matrix)
                    block_space.mapper.copy_local_to_global(coupling_matrices_in_out[(ii, jj)],
                                                            coupling_patterns_in_out[(ii, jj)],
                                                            ii, jj, system_matrix)
                    block_space.mapper.copy_local_to_global(coupling_matrices_out_in[(ii, jj)],
                                                            coupling_patterns_out_in[(ii, jj)],
                                                            jj, ii, system_matrix)

        op = DuneXTMatrixOperator(system_matrix)
        mats = np.full((grid.num_subdomains, grid.num_subdomains), None)
        for ii in range(grid.num_subdomains):
            for jj in range(ii, grid.num_subdomains):
                if ii == jj:
                    mats[ii, ii] = Matrix(block_space.local_space(ii).size(),
                                          block_space.local_space(ii).size(),
                                          local_patterns[ii])
                elif (ii, jj) in coupling_matrices_in_out:
                    mats[ii, jj] = Matrix(block_space.local_space(ii).size(),
                                          block_space.local_space(jj).size(),
                                          coupling_patterns_in_out[(ii, jj)])
                    mats[jj, ii] = Matrix(block_space.local_space(jj).size(),
                                          block_space.local_space(ii).size(),
                                          coupling_patterns_out_in[(ii, jj)])

        for ii in range(grid.num_subdomains):
            for jj in range(ii, grid.num_subdomains):
                if ii == jj:
                    mats[ii, ii].axpy(1.,  local_matrices[ii])
                    if ii in boundary_matrices:
                        mats[ii, ii].axpy(1.,  boundary_matrices[ii])
                elif (ii, jj) in coupling_matrices_in_out:
                    mats[ii, ii].axpy(1., coupling_matrices_in_in[(ii, jj)])
                    mats[jj, jj].axpy(1., coupling_matrices_out_out[(ii, jj)])
                    mats[ii, jj].axpy(1., coupling_matrices_in_out[(ii, jj)])
                    mats[jj, ii].axpy(1., coupling_matrices_out_in[(ii, jj)])

        ops = np.full((grid.num_subdomains, grid.num_subdomains), None)
        for (ii, jj), mat in np.ndenumerate(mats):
            ops[ii, jj] = DuneXTMatrixOperator(mat,
                                               source_id='domain_{}'.format(jj),
                                               range_id='domain_{}'.format(ii)) if mat else None
        block_op = BlockOperator(ops)

        rhs = VectorFunctional(op.range.make_array([rhs_vector]))
        rhss = []
        for ii in range(grid.num_subdomains):
            rhss.append(ops[ii, ii].range.make_array([local_vectors[ii]]))
        block_rhs = VectorFunctional(block_op.range.make_array(rhss))

        return op, block_op, rhs, block_rhs

    ops, block_ops, rhss, block_rhss = zip(*(discretize_lhs_for_lambda(l) for l in affine_lambda['functions']))
    global_rhs = rhss[0]
    block_rhs = block_rhss[0]

    lambda_coeffs = affine_lambda['coefficients']
    global_operator = LincombOperator(ops, lambda_coeffs)
    block_op = LincombOperator(block_ops, lambda_coeffs, name='lhs')

    operators = {}
    global_rt_space = make_rt_space(grid)
    subdomain_rt_spaces = [global_rt_space.restrict_to_dd_subdomain_part(grid, ii)
                           for ii in range(grid.num_subdomains)]

    def assemble_oswald_interpolation_error():
        oi_ops = [OswaldInterpolationErrorOperator(ii, block_op.source, grid, block_space)
                  for ii in range(grid.num_subdomains)]
        return BlockDiagonalOperator(oi_ops, name='oswald_interpolation_error')

    oi_op = assemble_oswald_interpolation_error()

    def assemble_flux_reconstruction(lambda_xi):
        fr_ops = [FluxReconstructionOperator(ii, block_op.source, grid, block_space, global_rt_space,
                                             subdomain_rt_spaces, lambda_xi, kappa)
                  for ii in range(grid.num_subdomains)]
        return BlockDiagonalOperator(fr_ops)

    fr_op = LincombOperator([assemble_flux_reconstruction(lambda_xi) for lambda_xi in affine_lambda['functions']],
                            lambda_coeffs, name='flux_reconstruction')

    spaces = block_op.source.subspaces

    local_divergence_operators = []
    local_l2_products = []
    local_domain_ops = []
    local_rt_ops = []
    local_oi_ops = []
    local_est_rhs_functionals = []
    local_elliptic_products = []

    # assemble local products
    for ii in range(grid.num_subdomains):
        local_dg_space = block_space.local_space(ii)
        # we want a larger pattern to allow for axpy with other matrices
        tmp_local_matrix = Matrix(local_dg_space.size(),
                                  local_dg_space.size(),
                                  local_dg_space.compute_pattern('face_and_volume'))
        local_energy_product_ops = []
        local_energy_product_coeffs = []
        for func, coeff in zip(affine_lambda['functions'], affine_lambda['coefficients']):
            local_energy_product_ops.append(make_elliptic_matrix_operator(
                func, kappa, tmp_local_matrix.copy(), local_dg_space, over_integrate=0))
            local_energy_product_coeffs.append(coeff)
            local_energy_product_ops.append(make_penalty_product_matrix_operator(
                grid, ii, local_all_dirichlet_boundary_info,
                local_dg_space,
                func, kappa, over_integrate=0))
            local_energy_product_coeffs.append(coeff)
        local_l2_product = make_l2_matrix_operator(tmp_local_matrix.copy(), local_dg_space)
        del tmp_local_matrix
        local_assembler = make_system_assembler(local_dg_space)
        for local_product_op in local_energy_product_ops:
            local_assembler.append(local_product_op)
        local_assembler.append(local_l2_product)
        local_assembler.assemble()
        local_energy_product_name = 'local_energy_dg_product_{}'.format(ii)
        local_energy_product = LincombOperator([DuneXTMatrixOperator(op.matrix(),
                                                                     source_id='domain_{}'.format(ii),
                                                                     range_id='domain_{}'.format(ii))
                                                for op in local_energy_product_ops],
                                               local_energy_product_coeffs,
                                               name=local_energy_product_name)
        operators[local_energy_product_name] = local_energy_product.assemble(mu_bar).with_(
                name=local_energy_product_name)
        local_l2_product_name = 'local_l2_product_{}'.format(ii)
        operators[local_l2_product_name] = DuneXTMatrixOperator(local_l2_product.matrix(),
                                                                source_id='domain_{}'.format(ii),
                                                                range_id='domain_{}'.format(ii))
        local_l2_products.append(operators[local_l2_product_name])
        # divergence operator
        local_rt_space = global_rt_space.restrict_to_dd_subdomain_part(grid, ii)
        local_div_op = make_divergence_matrix_operator_on_subdomain(grid, ii, local_dg_space, local_rt_space)
        local_div_op.assemble()
        local_divergence_operators.append(DuneXTMatrixOperator(local_div_op.matrix(),
                                          source_id='LOCALRT_{}'.format(ii),
                                          range_id='domain_{}'.format(ii),
                                          name='local_divergence_{}'.format(ii)))

        lrt_ops = np.full(grid.num_subdomains, None)
        neighborhood = grid.neighborhood_of(ii)
        for kk in neighborhood:
            component = grid.neighborhood_of(kk).index(ii)
            assert fr_op.range.subspaces[kk].subspaces[component].id == 'LOCALRT_{}'.format(ii)
            lrt_ops[kk] = BlockProjectionOperator(fr_op.range.subspaces[kk], component)
        lrt_op = BlockRowOperator(lrt_ops, source_spaces=fr_op.range.subspaces, name='localrt_{}'.format(ii))
        local_domain_ops.append(BlockProjectionOperator(block_op.source, ii))

        lrt_ops = np.full(grid.num_subdomains, None)
        neighborhood = grid.neighborhood_of(ii)
        for kk in neighborhood:
            component = grid.neighborhood_of(kk).index(ii)
            assert fr_op.range.subspaces[kk].subspaces[component].id == 'LOCALRT_{}'.format(ii)
            lrt_ops[kk] = BlockProjectionOperator(fr_op.range.subspaces[kk], component)
        lrt_op = BlockRowOperator(lrt_ops, source_spaces=fr_op.range.subspaces, name='localrt_{}'.format(ii))
        local_rt_ops.append(lrt_op)

        loi_ops = np.full(grid.num_subdomains, None)
        for kk in neighborhood:
            component = grid.neighborhood_of(kk).index(ii)
            assert oi_op.range.subspaces[kk].subspaces[component].id == 'domain_{}'.format(ii)
            loi_ops[kk] = BlockProjectionOperator(oi_op.range.subspaces[kk], component)
        loi_op = BlockRowOperator(loi_ops, source_spaces=oi_op.range.subspaces, name='localoi_{}'.format(ii))
        local_oi_ops.append(loi_op)

        est_rhs_vector  = Vector(block_space.local_space(ii).size())
        l2_functional = make_l2_volume_vector_functional(f, est_rhs_vector,
                                                         block_space.local_space(ii),
                                                         over_integrate=2)
        local_assembler = make_system_assembler(block_space.local_space(ii))
        local_assembler.append(l2_functional)
        local_assembler.assemble()
        local_est_rhs_functionals.append(VectorFunctional(spaces[ii].make_array([est_rhs_vector])))

        matrix = make_local_elliptic_matrix_operator(grid, ii,
                                                     block_space.local_space(ii),
                                                     lambda_bar, kappa)
        matrix.assemble()
        matrix = matrix.matrix()
        local_elliptic_products.append(DuneXTMatrixOperator(matrix,
                                                            range_id='domain_{}'.format(ii),
                                                            source_id='domain_{}'.format(ii)))


    # assemble error estimator
    for ii in range(grid.num_subdomains):

        neighborhood = grid.neighborhood_of(ii)

        def assemble_estimator_diffusive_flux_aa(lambda_xi, lambda_xi_prime):
            diffusive_flux_aa_product = make_diffusive_flux_aa_product(
                    grid, ii,
                    block_space.local_space(ii),
                    lambda_bar,
                    lambda_u=lambda_xi, lambda_v=lambda_xi_prime,
                    kappa=kappa,
                    over_integrate=2)
            subdomain_walker = make_subdomain_walker(grid, ii)
            subdomain_walker.append(diffusive_flux_aa_product)
            subdomain_walker.walk()
            matrix = DuneXTMatrixOperator(diffusive_flux_aa_product.matrix(),
                                          range_id='domain_{}'.format(ii),
                                          source_id='domain_{}'.format(ii))
            df_ops = np.full((grid.num_subdomains,) * 2, None)
            df_ops[ii, ii] = matrix
            return BlockOperator(df_ops, range_spaces=spaces, source_spaces=spaces)

        def assemble_estimator_diffusive_flux_bb():
            diffusive_flux_bb_product = make_diffusive_flux_bb_product(
                    grid, ii,
                    subdomain_rt_spaces[ii],
                    lambda_bar,
                    kappa=kappa,
                    over_integrate=2)
            subdomain_walker = make_subdomain_walker(grid, ii)
            subdomain_walker.append(diffusive_flux_bb_product)
            subdomain_walker.walk()
            matrix = DuneXTMatrixOperator(diffusive_flux_bb_product.matrix(),
                                          range_id='LOCALRT_{}'.format(ii),
                                          source_id='LOCALRT_{}'.format(ii))
            return Concatenation(local_rt_ops[ii].T, Concatenation(matrix, local_rt_ops[ii]),
                                 name='diffusive_flux_bb_{}'.format(ii))

        def assemble_estimator_diffusive_flux_ab(lambda_xi):
            diffusive_flux_ab_product = make_diffusive_flux_ab_product(
                    grid, ii,
                    range_space=block_space.local_space(ii),
                    source_space=subdomain_rt_spaces[ii],
                    lambda_range=lambda_xi,
                    lambda_hat=lambda_bar,
                    kappa=kappa,
                    over_integrate=2)
            subdomain_walker = make_subdomain_walker(grid, ii)
            subdomain_walker.append(diffusive_flux_ab_product)
            subdomain_walker.walk()
            matrix = DuneXTMatrixOperator(diffusive_flux_ab_product.matrix(),
                                          range_id='domain_{}'.format(ii),
                                          source_id='LOCALRT_{}'.format(ii))
            return Concatenation(local_domain_ops[ii].T, Concatenation(matrix, local_rt_ops[ii]))

        local_div = Concatenation(local_divergence_operators[ii], local_rt_ops[ii])
        local_oi = local_oi_ops[ii]

        operators['nc_{}'.format(ii)] = \
            Concatenation(local_oi.T, Concatenation(local_elliptic_products[ii], local_oi),
                          name='nonconformity_{}'.format(ii))

        operators['r1_{}'.format(ii)] = \
            Concatenation(local_est_rhs_functionals[ii], local_div, name='r1_{}'.format(ii))

        operators['r2_{}'.format(ii)] = \
            Concatenation(local_div.T, Concatenation(local_l2_products[ii], local_div), name='r2_{}'.format(ii))

        operators['df_aa_{}'.format(ii)] = LincombOperator(
            [assemble_estimator_diffusive_flux_aa(lambda_xi, lambda_xi_prime)
             for lambda_xi in affine_lambda['functions']
             for lambda_xi_prime in affine_lambda['functions']],
            [ProductParameterFunctional([c1, c2])
             for c1 in lambda_coeffs
             for c2 in lambda_coeffs],
            name='diffusive_flux_aa_{}'.format(ii))
        operators['df_bb_{}'.format(ii)] = assemble_estimator_diffusive_flux_bb()
        operators['df_ab_{}'.format(ii)] = LincombOperator(
            [assemble_estimator_diffusive_flux_ab(lambda_xi) for lambda_xi in affine_lambda['functions']],
            lambda_coeffs,
            name='diffusive_flux_ab_{}'.format(ii)
        )

    min_diffusion_evs = np.array([min_diffusion_eigenvalue(grid, ii, lambda_hat, kappa) for ii in
                                  range(grid.num_subdomains)])
    subdomain_diameters = np.array([subdomain_diameter(grid, ii) for ii in range(grid.num_subdomains)])
    local_eta_rf_squared = np.array([apply_l2_product(grid, ii, f, f, over_integrate=2) for ii in
                                     range(grid.num_subdomains)])
    estimator = Estimator(min_diffusion_evs, subdomain_diameters, local_eta_rf_squared, lambda_coeffs, mu_bar, mu_hat,
                          fr_op, oi_op)

    neighborhoods = [grid.neighborhood_of(ii) for ii in range(grid.num_subdomains)]
    local_boundary_info = make_subdomain_boundary_info(grid_and_problem_data['grid'],
                                                       {'type': 'xt.grid.boundaryinfo.alldirichlet'})
    d = DuneDiscretization(global_operator,
                           global_rhs,
                           neighborhoods,
                           (grid, local_boundary_info, affine_lambda, kappa, f, block_space),
                           block_op,
                           block_rhs,
                           visualizer=DuneGDTVisualizer(block_space),
                           operators=operators, estimator=estimator)
    d = d.with_(parameter_space=CubicParameterSpace(d.parameter_type, parameter_range[0], parameter_range[1]))

    return d, block_space

#!/usr/bin/env python
# import scipy.sparse
# from mybmat import mybmat
# scipy.sparse.bmat = mybmat

from itertools import product

import numpy as np

from pymor.basic import *
from pymor.core.exceptions import ExtensionError
from pymor.core.interfaces import ImmutableInterface
from pymor.operators.basic import OperatorBase
from pymor.operators.block import BlockOperator
from pymor.reductors.system import GenericRBSystemReductor

from dune.xt.la import IstlDenseVectorDouble as Vector
from dune.gdt import make_discrete_function


class EstimatorOperatorBase(OperatorBase):

    linear = True

    def __init__(self, subdomain, jj, kk, global_space, grid, block_space, global_rt_space, local_boundary_info,
                 lambda_hat, lambda_xi, lambda_xi_prime, kappa):
        self.range = global_space.subspaces[jj]
        self.source = global_space.subspaces[kk]
        self.global_space = global_space
        self.grid = grid
        self.block_space = block_space
        self.global_rt_space = global_rt_space
        self.subdomain = subdomain
        self.neighborhood = grid.neighborhood_of(subdomain)
        self.local_boundary_info = local_boundary_info
        self.lambda_hat = lambda_hat
        self.lambda_xi = lambda_xi
        self.lambda_xi_prime = lambda_xi_prime
        self.kappa = kappa
        self.jj = jj
        self.kk = kk

    def localize_to_subdomain_with_neighborhood_support(self, U, ss):
        assert len(U) == 1

        neighborhood_space = self.block_space.restricted_to_neighborhood(self.neighborhood)

        return make_discrete_function(
            neighborhood_space,
            neighborhood_space.project_onto_neighborhood(
                [U._list[0].impl if nn == ss else Vector(block_space.local_space(nn).size(), 0.)
                 for nn in self.neighborhood],
                self.neighborhood))

    def localize_to_subdomain_with_global_support(self, U, ss):
        assert len(U) == 1

        return make_discrete_function(
            self.block_space,
            self.block_space.project_onto_neighborhood(
                [U._list[0].impl if nn == ss else Vector(block_space.local_space(nn).size(), 0.)
                 for nn in range(self.grid.num_subdomains)],
                set([nn for nn in range(self.grid.num_subdomains)])
            )
        )

    def apply(self, U, mu=None):
        result = self.range.empty(reserve=len(U))
        for u_i in range(len(U)):
            result.append(self._apply(U[u_i], mu=mu))
        return result

    def apply2(self, V, U, mu=None):
        assert V in self.range and U in self.source
        result = np.empty((len(V), len(U)))
        for v_i in range(len(V)):
            for u_i in range(len(U)):
                result[v_i, u_i] = self._apply2(V[v_i], U[u_i], mu=mu)
        return result


class NonconformatyOperator(EstimatorOperatorBase):

    def apply(self, U, mu=None):
        raise NotImplementedError

    def _apply2(self, V, U, mu=None):
        from dune.gdt import make_ESV2007_nonconformity_product_dd_subdomain_part_dd_subdomain_oversampled_part \
            as make_local_nonconformity_product

        assert len(V) == 1 and len(U) == 1
        assert V in self.range and U in self.source

        subdomain_vh_with_neighborhood_support = self.localize_to_subdomain_with_neighborhood_support(V, self.jj)
        subdomain_uh_with_neighborhood_support = self.localize_to_subdomain_with_neighborhood_support(U, self.kk)
        local_eta_nc_squared = make_local_nonconformity_product(
            self.grid, self.subdomain, self.subdomain, self.local_boundary_info,
            self.lambda_hat, self.kappa,
            subdomain_vh_with_neighborhood_support,
            subdomain_uh_with_neighborhood_support,
            over_integrate=2).apply2()
        return np.array([[local_eta_nc_squared]])


class ResidualPartOperator(EstimatorOperatorBase):

    linear = True

    def apply(self, U, mu=None):
        raise NotImplementedError

    def _apply2(self, V, U, mu=None):
        from dune.gdt import (
            RS2017_apply_l2_product as apply_l2_product,
            apply_diffusive_flux_reconstruction_operator
        )

        assert len(V) == 1 and len(U) == 1
        assert V in self.source and U in self.range

        subdomain_vhs_with_global_support = self.localize_to_subdomain_with_global_support(V, self.jj)
        subdomain_uhs_with_global_support = self.localize_to_subdomain_with_global_support(U, self.kk)

        reconstructed_vh_jj_with_global_support = make_discrete_function(self.global_rt_space)
        apply_diffusive_flux_reconstruction_operator(
            self.lambda_xi_prime, self.kappa,
            subdomain_vhs_with_global_support,
            reconstructed_vh_jj_with_global_support)

        reconstructed_uh_kk_with_global_support = make_discrete_function(self.global_rt_space)
        apply_diffusive_flux_reconstruction_operator(
            self.lambda_xi_prime, self.kappa,
            subdomain_uhs_with_global_support,
            reconstructed_uh_kk_with_global_support)

        return apply_l2_product(
            self.grid, self.subdomain,
            reconstructed_vh_jj_with_global_support.divergence(),
            reconstructed_uh_kk_with_global_support.divergence(),
            over_integrate=2
        )


class ResidualPartFunctional(EstimatorOperatorBase):

    linear = True

    def __init__(self, f, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.range = NumpyVectorSpace(1)
        self.f = f

    def _apply(self, U, mu=None):
        from dune.gdt import (
            RS2017_apply_l2_product as apply_l2_product,
            apply_diffusive_flux_reconstruction_operator
        )

        assert len(U) == 1
        assert U in self.source

        subdomain_uhs_with_global_support = self.localize_to_subdomain_with_global_support(U, self.kk)

        reconstructed_uh_jj_with_global_support = make_discrete_function(self.global_rt_space)
        apply_diffusive_flux_reconstruction_operator(
            self.lambda_xi_prime, self.kappa,
            subdomain_uhs_with_global_support,
            reconstructed_uh_jj_with_global_support)
        result = apply_l2_product(
            self.grid, self.subdomain,
            self.f,
            reconstructed_uh_jj_with_global_support.divergence(),
            over_integrate=2
        )
        return self.range.from_data(np.array([[result]]))


class DiffusiveFluxOperator(EstimatorOperatorBase):

    linear = True

    def apply(self, U, mu=None):
        raise NotImplementedError

    def _apply2(self, V, U, mu=None):
        from dune.gdt import (
            RS2017_diffusive_flux_indicator_apply_aa_product as apply_diffusive_flux_aa_product,
            RS2017_diffusive_flux_indicator_apply_ab_product as apply_diffusive_flux_ab_product,
            RS2017_diffusive_flux_indicator_apply_bb_product as apply_diffusive_flux_bb_product,
            apply_diffusive_flux_reconstruction_operator
        )

        assert len(V) == 1 and len(U) == 1
        assert V in self.range and U in self.source

        subdomain_vhs_with_global_support = self.localize_to_subdomain_with_global_support(V, self.jj)
        subdomain_uhs_with_global_support = self.localize_to_subdomain_with_global_support(U, self.kk)

        reconstructed_vh_jj_with_global_support = make_discrete_function(self.global_rt_space)
        apply_diffusive_flux_reconstruction_operator(
            self.lambda_xi, self.kappa,
            subdomain_vhs_with_global_support,
            reconstructed_vh_jj_with_global_support)

        reconstructed_uh_kk_with_global_support = make_discrete_function(self.global_rt_space)
        apply_diffusive_flux_reconstruction_operator(
            self.lambda_xi_prime, self.kappa,
            subdomain_uhs_with_global_support,
            reconstructed_uh_kk_with_global_support)

        if self.subdomain == self.jj:
            subdomain_vh = make_discrete_function(self.block_space.local_space(self.subdomain), V._list[0].impl)
        if self.subdomain == self.kk:
            subdomain_uh = make_discrete_function(self.block_space.local_space(self.subdomain), U._list[0].impl)

        local_eta_df_squared = 0

        if self.subdomain == self.jj == self.kk:
            local_eta_df_squared += apply_diffusive_flux_aa_product(
                self.grid, self.subdomain,
                self.lambda_hat, lambda_u=self.lambda_xi, lambda_v=self.lambda_xi_prime,
                kappa=self.kappa,
                u=subdomain_uh,
                v=subdomain_vh,
                over_integrate=2)

        if self.subdomain == self.jj:
            local_eta_df_squared += apply_diffusive_flux_ab_product(
                self.grid, self.subdomain,
                self.lambda_hat,
                lambda_u=self.lambda_xi_prime,
                kappa=self.kappa,
                u=subdomain_vh,
                reconstructed_v=reconstructed_uh_kk_with_global_support,
                over_integrate=2)

        if self.subdomain == self.kk:
            local_eta_df_squared += apply_diffusive_flux_ab_product(
                self.grid, self.subdomain,
                self.lambda_hat,
                lambda_u=self.lambda_xi_prime,
                kappa=self.kappa,
                u=subdomain_uh,
                reconstructed_v=reconstructed_vh_jj_with_global_support,
                over_integrate=2)

        local_eta_df_squared += apply_diffusive_flux_bb_product(
            self.grid, self.subdomain,
            self.lambda_hat, self.kappa,
            reconstructed_vh_jj_with_global_support,
            reconstructed_uh_kk_with_global_support,
            over_integrate=2)

        return np.array([[local_eta_df_squared]])


class Estimator(ImmutableInterface):

    def __init__(self, min_diffusion_evs, subdomain_diameters, local_eta_rf_squared):
        self.min_diffusion_evs = min_diffusion_evs
        self.subdomain_diameters = subdomain_diameters
        self.local_eta_rf_squared = local_eta_rf_squared

    def estimate(self, U, mu, discretization, decompose=False):
        d = discretization

        alpha_mu_mu_bar = 1.
        gamma_mu_mu_bar = 1.
        alpha_mu_mu_hat = 1.

        local_eta_nc = np.zeros(grid.num_subdomains)
        local_eta_r = np.zeros(grid.num_subdomains)
        local_eta_df = np.zeros(grid.num_subdomains)

        for ii in range(grid.num_subdomains):
            local_eta_nc[ii] = d.operators['nc_{}'.format(ii)].apply2(U, U)
            local_eta_r[ii] += self.local_eta_rf_squared[ii]
            local_eta_r[ii] -= 2*d.operators['r1_{}'.format(ii)].apply(U).data
            local_eta_r[ii] += d.operators['r2_{}'.format(ii)].apply2(U, U)
            local_eta_df[ii] += d.operators['df_{}'.format(ii)].apply2(U, U)

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
            return eta, (local_eta_nc, local_eta_r, local_eta_df)
        else:
            return eta


class DuneDiscretization(StationaryDiscretization):

    def _solve(self, mu):
        if not self.logging_disabled:
            self.logger.info('Solving {} for {} ...'.format(self.name, mu))

        return self.solution_space.from_data(
            self.operators['global_op'].apply_inverse(self.operators['global_rhs'].as_vector(mu=mu),
                                                      mu=mu).data
        )

    def as_generic_type(self):
        ops = dict(self.operators)
        del ops['operator']
        del ops['rhs']
        del ops['global_op']
        del ops['global_rhs']
        return StationaryDiscretization(self.operator, self.rhs, operators=ops, parameter_space=self.parameter_space)

    def visualize(self, U, *args, **kwargs):
        self.visualizer.visualize(self.unblock(U), self, *args, **kwargs)

    def unblock(self, U):
        return self.operators['global_op'].source.from_data(U.data)


def discretize():
    from dune.xt.common import init_logger, init_mpi
    init_mpi()
    init_logger()

    from dune.xt.functions import (
        make_checkerboard_function_1x1,
        make_constant_function_1x1,
        make_constant_function_2x2,
        make_expression_function_1x1
    )
    from dune.xt.grid import (
        make_cube_dd_subdomains_grid__2d_simplex_aluconform as make_grid,
        make_boundary_info_on_dd_subdomain_layer as make_subdomain_boundary_info,
        make_boundary_info_on_dd_subdomain_boundary_layer as make_boundary_info,
        make_apply_on_dirichlet_intersections_dd_subdomain_boundary_part as make_apply_on_dirichlet_intersections
    )
    from dune.xt.la import (
        IstlRowMajorSparseMatrixDouble as Matrix,
        IstlDenseVectorDouble as Vector,
        SparsityPatternDefault
    )
    from dune.gdt import (
        make_block_dg_dd_subdomain_part_to_1x1_fem_p1_space as make_block_space,
        make_elliptic_swipdg_affine_factor_matrix_operator as make_elliptic_swipdg_matrix_operator,
        make_l2_volume_vector_functional,
        make_local_elliptic_swipdg_affine_factor_inner_integral_operator_1x1_p1_dg_fem_space_dd_subdomain_coupling_intersection as make_local_elliptic_swipdg_coupling_operator,  # NOQA
        make_local_elliptic_swipdg_affine_factor_boundary_integral_operator_1x1_p1_dg_fem_space_dd_subdomain_coupling_intersection as make_local_elliptic_swipdg_boundary_operator,  # NOQA
        make_system_assembler,
        make_rt_leaf_view_to_2x1_pdelab_p0_space as make_rt_space,
        RS2017_residual_indicator_min_diffusion_eigenvalue as min_diffusion_eigenvalue,
        RS2017_residual_indicator_subdomain_diameter as subdomain_diameter,
    )

    inner_boundary_id = 18446744073709551573
    grid = make_grid(lower_left=[-1, -1], upper_right=[1, 1], num_elements=[4, 4], num_refinements=2,
                     num_partitions=[2, 2], num_oversampling_layers=4, inner_boundary_segment_index=inner_boundary_id)

    all_dirichlet_boundary_info = make_boundary_info(grid, {'type': 'xt.grid.boundaryinfo.alldirichlet'})
    all_neumann_boundary_info = make_subdomain_boundary_info(grid, {'type': 'xt.grid.boundaryinfo.allneumann'})

    XBLOCKS = YBLOCKS = 2

    def diffusion_function_factory(ix, iy):
        values = [[0.]]*(YBLOCKS*XBLOCKS)
        values[ix + XBLOCKS*iy] = [1.]
        return make_checkerboard_function_1x1(grid_provider=grid, lower_left=[-1, -1], upper_right=[1, 1],
                                              num_elements=[XBLOCKS, YBLOCKS],
                                              values=values, name='diffusion_{}_{}'.format(ix, iy))

    diffusion_functions = [diffusion_function_factory(ix, iy)
                           for ix, iy in product(range(XBLOCKS), range(YBLOCKS))]

    kappa = make_constant_function_2x2(grid, [[1., 0.], [0., 1.]], name='diffusion')
    # f = make_expression_function_1x1(grid, 'x', '1', order=0, name='force')
    f = make_expression_function_1x1(grid, 'x', '0.5*pi*pi*cos(0.5*pi*x[0])*cos(0.5*pi*x[1])', order=2, name='f')

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
            ipdg_operator = make_elliptic_swipdg_matrix_operator(lambda_, kappa, all_neumann_boundary_info,
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
        apply_on_dirichlet_intersections = make_apply_on_dirichlet_intersections(all_dirichlet_boundary_info)

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

        from pymor.bindings.dunext import DuneXTMatrixOperator

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

    ops, block_ops, rhss, block_rhss = zip(*(discretize_lhs_for_lambda(l) for l in diffusion_functions))
    rhs = rhss[0]
    block_rhs = block_rhss[0]

    coefficients = [ProjectionParameterFunctional(component_name='diffusion',
                                                  component_shape=(YBLOCKS, XBLOCKS),
                                                  coordinates=(YBLOCKS - y - 1, x))
                    for x in range(XBLOCKS) for y in range(YBLOCKS)]
    op = LincombOperator(ops, coefficients)
    block_op = LincombOperator(block_ops, coefficients, name='lhs')

    local_boundary_info = make_subdomain_boundary_info(
        grid,
        {'type': 'xt.grid.boundaryinfo.boundarysegmentindexbased',
         'default': 'dirichlet',
         'neumann': '[{} {}]'.format(inner_boundary_id, inner_boundary_id+1)})

    lambda_hat = make_constant_function_1x1(grid, 1.0, name='lambda')
    lambda_xi = lambda_hat
    lambda_xi_prime = lambda_hat
    operators = {'global_op': op, 'global_rhs': rhs}
    global_rt_space = make_rt_space(grid)

    for ii in range(grid.num_subdomains):
        nc_ops = np.full((grid.num_subdomains,) * 2, None)
        r1_ops = np.full((1, grid.num_subdomains,), None)
        r2_ops = np.full((grid.num_subdomains,) * 2, None)
        df_ops = np.full((grid.num_subdomains,) * 2, None)
        neighborhood = grid.neighborhood_of(ii)
        for jj in neighborhood:
            r1_ops[0, jj] = ResidualPartFunctional(f, ii, jj, jj, block_op.source, grid, block_space, global_rt_space,
                                                   local_boundary_info, lambda_hat, lambda_xi, lambda_xi_prime, kappa)
            for kk in neighborhood:
                nc_ops[jj, kk] = NonconformatyOperator(ii, jj, kk, block_op.source, grid, block_space, global_rt_space,
                                                       local_boundary_info, lambda_hat, lambda_xi, lambda_xi_prime, kappa)
                r2_ops[jj, kk] = ResidualPartOperator(ii, jj, kk, block_op.source, grid, block_space, global_rt_space,
                                                      local_boundary_info, lambda_hat, lambda_xi, lambda_xi_prime, kappa)
                df_ops[jj, kk] = DiffusiveFluxOperator(ii, jj, kk, block_op.source, grid, block_space, global_rt_space,
                                                       local_boundary_info, lambda_hat, lambda_xi, lambda_xi_prime, kappa)
        operators['nc_{}'.format(ii)] = BlockOperator(nc_ops,
                                                      range_spaces=block_op.range.subspaces,
                                                      source_spaces=block_op.source.subspaces,
                                                      name='nonconformity_{}'.format(ii))
        operators['r1_{}'.format(ii)] = BlockOperator(r1_ops,
                                                      source_spaces=block_op.source.subspaces,
                                                      name='residual_functional_{}'.format(ii))
        operators['r2_{}'.format(ii)] = BlockOperator(r2_ops,
                                                      range_spaces=block_op.range.subspaces,
                                                      source_spaces=block_op.source.subspaces,
                                                      name='residual_{}'.format(ii))
        operators['df_{}'.format(ii)] = BlockOperator(df_ops,
                                                      range_spaces=block_op.range.subspaces,
                                                      source_spaces=block_op.source.subspaces,
                                                      name='diffusive_flux_{}'.format(ii))

    from dune.gdt import RS2017_apply_l2_product as apply_l2_product
    min_diffusion_evs = np.array([min_diffusion_eigenvalue(grid, ii, lambda_hat, kappa) for ii in
                                  range(grid.num_subdomains)])
    subdomain_diameters = np.array([subdomain_diameter(grid, ii) for ii in range(grid.num_subdomains)])
    local_eta_rf_squared = np.array([apply_l2_product(grid, ii, f, f, over_integrate=2) for ii in
                                     range(grid.num_subdomains)])
    estimator = Estimator(min_diffusion_evs, subdomain_diameters, local_eta_rf_squared)

    from pymor.bindings.dunegdt import DuneGDTVisualizer

    d = DuneDiscretization(block_op, block_rhs, visualizer=DuneGDTVisualizer(block_space),
                           operators=operators, estimator=estimator)
    d = d.with_(parameter_space=CubicParameterSpace(d.parameter_type, 0.1, 1.))

    return d, grid, block_space, f, kappa


if __name__ == '__main__':
    d, grid, block_space, f, kappa = discretize()

    U = d.solution_space.empty()
    reductor = GenericRBSystemReductor(d)
    for mu in d.parameter_space.sample_uniformly(2):
        snapshot = d.solve(mu)
        U.append(snapshot)
        try:
            reductor.extend_basis(snapshot)
        except ExtensionError:
            break
    d.visualize(U, filename='U')
    rd = reductor.reduce()

    u = rd.solution_space.empty()
    for mu in d.parameter_space.sample_uniformly(2):
        u.append(rd.solve(mu))
    UU = reductor.reconstruct(u)
    print((U - UU).l2_norm() / U.l2_norm())

    U = d.solve([1., 1., 1., 1.])

    print('estimating error ', end='', flush=True)

    eta, (local_eta_nc, local_eta_r, local_eta_df) = d.estimate(U, decompose=True)

    print('')
    print('  nonconformity indicator:  {} (should be 1.66e-01)'.format(np.linalg.norm(local_eta_nc)))
    print('  residual indicator:       {} (should be 2.89e-01)'.format(np.linalg.norm(local_eta_r)))
    print('  diffusive flux indicator: {} (should be 3.55e-01)'.format(np.linalg.norm(local_eta_df)))
    print('  estimated error:          {}'.format(eta))

    rd = rd.with_(estimator=d.estimator)
    u = rd.solve([1., 1., 1., 1.])

    print('estimating reduced error ', end='', flush=True)

    eta, (local_eta_nc, local_eta_r, local_eta_df) = rd.estimate(u, decompose=True)

    print('')
    print('  nonconformity indicator:  {} (should be 1.66e-01)'.format(np.linalg.norm(local_eta_nc)))
    print('  residual indicator:       {} (should be 2.89e-01)'.format(np.linalg.norm(local_eta_r)))
    print('  diffusive flux indicator: {} (should be 3.55e-01)'.format(np.linalg.norm(local_eta_df)))
    print('  estimated error:          {}'.format(eta))

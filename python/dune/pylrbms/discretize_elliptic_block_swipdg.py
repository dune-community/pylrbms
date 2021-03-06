import numpy as np
from mpi4py import MPI

from dune.xt.grid.walker import (
    make_apply_on_dirichlet_intersections,
    make_walker_on_dd_subdomain_view as make_subdomain_walker
)
from dune.xt.grid.boundaryinfo import make_boundary_info_on_dd_subdomain_layer as make_subdomain_boundary_info
from dune.xt.functions import make_expression_function_1x1
from dune.xt.la import (
    IstlDenseVectorDouble as Vector,
    IstlRowMajorSparseMatrixDouble as Matrix,
    SparsityPatternDefault,
)

import dune.gdt

from dune.gdt.spaces import make_rt_space
from dune.gdt.spaces import make_block_dg_space

from dune.gdt.__local_elliptic_ipdg_operators import (
    make_local_elliptic_swipdg_affine_factor_boundary_integral_operator_1x1_p1_dg_gdt_space_dd_subdomain_coupling_intersection as make_local_elliptic_swipdg_boundary_operator,
    make_local_elliptic_swipdg_affine_factor_inner_integral_operator_1x1_p1_dg_gdt_space_dd_subdomain_coupling_intersection as make_local_elliptic_swipdg_coupling_operator,  # NOQA
)
from dune.gdt.__discretefunction import make_discrete_function
from dune.gdt.__operators_l2 import make_l2_matrix_operator
from dune.gdt.__functionals_l2 import make_l2_volume_vector_functional
from dune.gdt.__operators_elliptic import make_elliptic_matrix_operator_istl_row_major_sparse_matrix_double as make_elliptic_matrix_operator
from dune.gdt.__operators_elliptic_ipdg import make_elliptic_swipdg_affine_factor_matrix_operator as make_elliptic_swipdg_matrix_operator
from dune.gdt.__operators_oswaldinterpolation import apply_oswald_interpolation_operator
from dune.gdt.__operators_RS2017 import (
    RS2017_apply_diffusive_flux_reconstruction_in_neighborhood as apply_diffusive_flux_reconstruction_in_neighborhood,
    RS2017_make_diffusive_flux_aa_product_matrix_operator_on_subdomain as make_diffusive_flux_aa_product,
    RS2017_make_diffusive_flux_ab_product_matrix_operator_on_subdomain as make_diffusive_flux_ab_product,
    RS2017_make_diffusive_flux_bb_product_matrix_operator_on_subdomain as make_diffusive_flux_bb_product,
    RS2017_apply_l2_product as apply_l2_product,
    RS2017_make_elliptic_matrix_operator_on_subdomain as make_local_elliptic_matrix_operator,
    RS2017_make_elliptic_swipdg_matrix_operator_on_neighborhood as make_elliptic_swipdg_matrix_operator_on_neighborhood,
    RS2017_make_elliptic_swipdg_vector_functional_on_neighborhood as make_elliptic_swipdg_vector_functional_on_neighborhood,  # NOQA
    RS2017_make_l2_vector_functional_on_neighborhood as make_l2_vector_functional_on_neighborhood,
    RS2017_make_neighborhood_system_assembler as make_neighborhood_system_assembler,
    RS2017_make_penalty_product_matrix_operator_on_subdomain as make_penalty_product_matrix_operator,
    RS2017_residual_indicator_min_diffusion_eigenvalue as min_diffusion_eigenvalue,
    RS2017_residual_indicator_subdomain_diameter as subdomain_diameter,
    RS2017_make_divergence_matrix_operator_on_subdomain as make_divergence_matrix_operator_on_subdomain,
)

from dune.gdt.__assembler import  make_system_assembler

from pymor.bindings.dunegdt import DuneGDTVisualizer
from pymor.bindings.dunext import DuneXTMatrixOperator, DuneXTVectorSpace
from pymor.core.interfaces import ImmutableInterface
from pymor.core.logger import getLogger
from pymor.discretizations.basic import StationaryDiscretization
from pymor.operators.basic import OperatorBase
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.block import BlockOperator, BlockDiagonalOperator, BlockProjectionOperator, BlockRowOperator
from pymor.operators.constructions import LincombOperator, VectorFunctional, Concatenation
from pymor.parameters.functionals import ProductParameterFunctional
from pymor.parameters.spaces import CubicParameterSpace
from pymor.vectorarrays.block import BlockVectorSpace


from dune.pylrbms.estimators import EllipticEstimator

def _get_subdomains(grid):
    ls = grid.subdomains_on_rank
    nls = len(ls)
    ngs= grid.num_subdomains
    return ls, nls, ngs
    
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
                        [U._list[u_i].impl if nn == self.subdomain else
                         Vector(self.block_space.local_space(nn).size(), 0.)
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
                result._blocks[i_ii]._list[0].data[:] -= \
                    np.frombuffer(interpolated_u_vector)[offsets[ind]:offsets[ind+1]]
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
        self.range = BlockVectorSpace(
            [DuneXTVectorSpace(vector_type, subdomain_rt_spaces[ii].size(), 'LOCALRT_' + str(ii))
             for ii in self.grid.neighborhood_of(subdomain)],
            'RT_{}'.format(subdomain)
        )

    def apply(self, U, mu=None):
        assert U in self.source
        result = self.range.empty(reserve=len(U))
        local_subdomains, num_local_subdomains, num_global_subdomains = _get_subdomains(self.grid)
        for u_i in range(len(U)):
            subdomain_uhs_with_global_support = \
                make_discrete_function(
                    self.block_space,
                    self.block_space.project_onto_neighborhood(
                        [U._list[u_i].impl if nn == self.subdomain else
                         Vector(self.block_space.local_space(nn).size(), 0.)
                         for nn in range(num_global_subdomains)],
                        [nn for nn in range(num_global_subdomains)]
                    )
                )

            reconstructed_uh_kk_with_global_support = make_discrete_function(self.global_rt_space)
            apply_diffusive_flux_reconstruction_in_neighborhood(
                self.grid, self.subdomain,
                self.lambda_xi, self.kappa,
                subdomain_uhs_with_global_support,
                reconstructed_uh_kk_with_global_support)

            blocks = [s.make_array([self.subdomain_rt_spaces[ii].restrict(
                                        reconstructed_uh_kk_with_global_support.vector_copy())])  # NOQA
                      for s, ii in zip(self.range.subspaces, self.grid.neighborhood_of(self.subdomain))]
            result.append(self.range.make_array(blocks))

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
                func = make_expression_function_1x1(self.grid, 'x', expression, order=2)
                dune_project(func, tmp_discrete_function)
                U.append(local_space.make_array([tmp_discrete_function.vector_copy()]))

        return U


class DuneDiscretization(DuneDiscretizationBase, StationaryDiscretization):

    def __init__(self, global_operator, global_rhs,
                 neighborhoods,
                 enrichment_data,  # = grid, local_boundary_info, affine_lambda, kappa, f, block_space
                 operator, rhs,
                 products=None, operators=None,
                 parameter_space=None, estimator=None, visualizer=None, cache_region=None, name=None, data=None):
        super().__init__(operator, rhs, products=products, operators=operators,
                         parameter_space=parameter_space, estimator=estimator, visualizer=visualizer,
                         cache_region=cache_region, name=name)
        self.global_operator, self.global_rhs, self.neighborhoods, self.enrichment_data = \
            global_operator, global_rhs, neighborhoods, enrichment_data
        self.data = data
        self.block_operator=operator

    def _solve(self, mu, inverse_options=None):
        if not self.logging_disabled:
            self.logger.info('Solving {} for {} ...'.format(self.name, mu))

        return self.solution_space.from_data(
            self.global_operator.apply_inverse(self.global_rhs.as_vector(mu=mu), mu=mu, inverse_options=inverse_options).data
        )

    def solve_for_local_correction(self, subdomain, Us, mu=None, inverse_options=None):
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

        # We don't have any boundary treatment right now. Things will probably
        # break in multiple ways in case of non-trivial boundary conditions,
        # so we can comment this out for now ..

        # for lambda_ in affine_lambda['functions']:
        #     funcs.append(make_elliptic_swipdg_vector_functional_on_neighborhood(
        #         grid, subdomain, local_boundary_info,
        #         neighborhood_space,
        #         current_solution, lambda_, kappa,
        #         over_integrate=0))
        # funcs_coeffs = affine_lambda['coefficients'].copy()
        funcs.append(make_l2_vector_functional_on_neighborhood(
            grid, subdomain,
            neighborhood_space,
            f,
            over_integrate=2))
        # funcs_coeffs.append(1.)
        funcs_coeffs = [1]
        #   assemble in one grid walk
        neighborhood_assembler = make_neighborhood_system_assembler(grid, subdomain, neighborhood_space)
        for op in ops:
            neighborhood_assembler.append(op)
        for func in funcs:
            neighborhood_assembler.append(func)
        neighborhood_assembler.assemble()
        # solve
        local_space_id = self.solution_space.subspaces[subdomain].id
        # lhs = LincombOperator([DuneXTMatrixOperator(o.matrix(), source_id=local_space_id, range_id=local_space_id)
        #                        for o in ops],
        #                       ops_coeffs)

        cols, rows = ops[0].matrix().cols, ops[0].matrix().rows
        assert cols == rows

        simple = False

        if simple:
            from scipy.sparse import coo_matrix
            eye = coo_matrix(np.eye(rows, cols))
            lhs = NumpyMatrixOperator(eye, source_id=local_space_id, range_id=local_space_id)
            rhs = VectorFunctional(lhs.range.make_array(np.ones(rows)))
            correction = lhs.apply_inverse(rhs.as_source_array(mu), mu=mu, inverse_options=None)
            localized_corrections_as_np = correction
        else:
            lhs = LincombOperator([DuneXTMatrixOperator(o.matrix(), source_id=local_space_id, range_id=local_space_id)
                                            for o in ops],ops_coeffs)
            rhs = LincombOperator([VectorFunctional(lhs.range.make_array([v.vector()])) for v in funcs], funcs_coeffs)
            correction = lhs.apply_inverse(rhs.as_source_array(mu), mu=mu, inverse_options=inverse_options)
            # correction = rhs.as_source_array(mu)

        assert len(correction) == 1
        # restrict to subdomain
        local_sizes = [block_space.local_space(nn).size() for nn in neighborhood]
        local_starts = [int(np.sum(local_sizes[:nn])) for nn in range(len(local_sizes))]
        local_starts.append(neighborhood_space.mapper.size)
        localized_corrections_as_np = np.array(correction._list[0].impl, copy=False)
        localized_corrections_as_np = [localized_corrections_as_np[local_starts[nn]:local_starts[nn+1]]
                                       for nn in range(len(local_sizes))]
        subdomain_index_in_neighborhood = np.where(np.array(list(neighborhood)) == subdomain)[0]
        assert len(subdomain_index_in_neighborhood) == 1
        subdomain_index_in_neighborhood = subdomain_index_in_neighborhood[0]
        subdomain_correction = Vector(local_sizes[subdomain_index_in_neighborhood], 0.)
        subdomain_correction_as_np = np.array(subdomain_correction, copy=False)
        subdomain_correction_as_np[:] = localized_corrections_as_np[subdomain_index_in_neighborhood][:]
        return self.solution_space.subspaces[subdomain].make_array([subdomain_correction])


def assemble_estimator_diffusive_flux_aa(lambda_xi, lambda_xi_prime, grid, ii, block_space, lambda_hat, kappa, solution_space):
    local_subdomains, num_local_subdomains, num_global_subdomains = _get_subdomains(grid)
    diffusive_flux_aa_product = make_diffusive_flux_aa_product(
        grid, ii,
        block_space.local_space(ii),
        lambda_hat,
        lambda_u=lambda_xi, lambda_v=lambda_xi_prime,
        kappa=kappa,
        over_integrate=2
    )
    subdomain_walker = make_subdomain_walker(grid, ii)
    subdomain_walker.append(diffusive_flux_aa_product)
    subdomain_walker.walk()
    # , block_space.local_space(ii).dof_communicator,
    matrix = DuneXTMatrixOperator(diffusive_flux_aa_product.matrix(),
                                  range_id='domain_{}'.format(ii),
                                  source_id='domain_{}'.format(ii))
    df_ops = np.full((num_global_subdomains,) * 2, None)
    df_ops[ii, ii] = matrix
    return BlockOperator(df_ops, range_spaces=solution_space.subspaces, source_spaces=solution_space.subspaces)


def assemble_estimator_diffusive_flux_bb(grid, ii, subdomain_rt_spaces, lambda_hat, kappa, local_rt_projection):
    diffusive_flux_bb_product = make_diffusive_flux_bb_product(
        grid, ii,
        subdomain_rt_spaces[ii],
        lambda_hat,
        kappa=kappa,
        over_integrate=2
    )
    subdomain_walker = make_subdomain_walker(grid, ii)
    subdomain_walker.append(diffusive_flux_bb_product)
    subdomain_walker.walk()
    # subdomain_rt_spaces[ii].dof_communicator,
    matrix = DuneXTMatrixOperator(diffusive_flux_bb_product.matrix(),
                                  range_id='LOCALRT_{}'.format(ii),
                                  source_id='LOCALRT_{}'.format(ii))
    return Concatenation([local_rt_projection.T, matrix, local_rt_projection],
                         name='diffusive_flux_bb_{}'.format(ii))


def assemble_estimator_diffusive_flux_ab(lambda_xi, grid, ii, block_space, subdomain_rt_spaces, lambda_hat, kappa,
                                         local_rt_projection, local_projection):
    diffusive_flux_ab_product = make_diffusive_flux_ab_product(
        grid, ii,
        range_space=block_space.local_space(ii),
        source_space=subdomain_rt_spaces[ii],
        lambda_range=lambda_xi,
        lambda_hat=lambda_hat,
        kappa=kappa,
        over_integrate=2
    )
    subdomain_walker = make_subdomain_walker(grid, ii)
    subdomain_walker.append(diffusive_flux_ab_product)
    subdomain_walker.walk()
    # subdomain_rt_spaces[ii].dof_communicator,
    matrix = DuneXTMatrixOperator(diffusive_flux_ab_product.matrix(),
                                  range_id='domain_{}'.format(ii),
                                  source_id='LOCALRT_{}'.format(ii))
    return Concatenation([local_projection.T, matrix, local_rt_projection])


def discretize_lhs(lambda_func, grid, block_space, local_patterns, boundary_patterns, coupling_matrices, kappa,
                   local_all_neumann_boundary_info, boundary_info, coupling_patterns, solver_options):
    logger = getLogger('discretize_lhs')
    logger.debug('...')
    local_subdomains, num_local_subdomains, num_global_subdomains = _get_subdomains(grid)
    local_matrices = [None]*num_global_subdomains
    boundary_matrices = {}
    logger.debug('discretize lhs coupling matrices ...')
    for ii in range(num_global_subdomains):
        local_matrices[ii] = Matrix(block_space.local_space(ii).size(),
                                    block_space.local_space(ii).size(),
                                    local_patterns[ii])
        if ii in grid.boundary_subdomains():
            boundary_matrices[ii] = Matrix(block_space.local_space(ii).size(),
                                           block_space.local_space(ii).size(),
                                           boundary_patterns[ii])

    logger.debug('discretize lhs ipdg ops ...')
    for ii in range(num_global_subdomains):
        ss = block_space.local_space(ii)
        ll = local_matrices[ii]
        # the operator itself is never used again, but the matrices it assembled are
        ipdg_operator = make_elliptic_swipdg_matrix_operator(lambda_func, kappa, local_all_neumann_boundary_info,
                                                             ll,
                                                             ss, over_integrate=2)
        ipdg_operator.assemble(False)

    logger.debug('discretize lhs ops ...')
    local_ipdg_coupling_operator = make_local_elliptic_swipdg_coupling_operator(lambda_func, kappa)

    def assemble_coupling_contributions(subdomain, neighboring_subdomain):
        coupling_assembler = block_space.coupling_assembler(subdomain, neighboring_subdomain)
        coupling_assembler.append(local_ipdg_coupling_operator,
                                  coupling_matrices['in_in'][(subdomain, neighboring_subdomain)],
                                  coupling_matrices['out_out'][(subdomain, neighboring_subdomain)],
                                  coupling_matrices['in_out'][(subdomain, neighboring_subdomain)],
                                  coupling_matrices['out_in'][(subdomain, neighboring_subdomain)])
        coupling_assembler.assemble()

    for ii in range(num_global_subdomains):
        for jj in grid.neighboring_subdomains(ii):
            if ii < jj:  # Assemble primally (visit each coupling only once).
                assemble_coupling_contributions(ii, jj)

    logger.debug('discretize lhs boundary ...')
    local_ipdg_boundary_operator = make_local_elliptic_swipdg_boundary_operator(lambda_func, kappa)
    apply_on_dirichlet_intersections = make_apply_on_dirichlet_intersections(boundary_info, grid=grid, layer='dd_subdomain_boundary_view')

    def assemble_boundary_contributions(subdomain):
        boundary_assembler = block_space.boundary_assembler(subdomain)
        boundary_assembler.append(local_ipdg_boundary_operator,
                                  boundary_matrices[subdomain],
                                  apply_on_dirichlet_intersections)
        boundary_assembler.assemble()

    for ii in grid.boundary_subdomains():
        assemble_boundary_contributions(ii)

    logger.debug('discretize lhs global contributions ...')
    global_pattern = SparsityPatternDefault(block_space.mapper.size)
    for ii in range(num_global_subdomains):
        block_space.mapper.copy_local_to_global(local_patterns[ii], ii, global_pattern)
        if ii in grid.boundary_subdomains():
            block_space.mapper.copy_local_to_global(boundary_patterns[ii], ii, global_pattern)
        for jj in grid.neighboring_subdomains(ii):
            if ii < jj:  # Assemble primally (visit each coupling only once).
                block_space.mapper.copy_local_to_global(coupling_patterns['in_in'][(ii, jj)], ii, ii, global_pattern)
                block_space.mapper.copy_local_to_global(coupling_patterns['out_out'][(ii, jj)], jj, jj, global_pattern)
                block_space.mapper.copy_local_to_global(coupling_patterns['in_out'][(ii, jj)], ii, jj, global_pattern)
                block_space.mapper.copy_local_to_global(coupling_patterns['out_in'][(ii, jj)], jj, ii, global_pattern)

    system_matrix = Matrix(block_space.mapper.size, block_space.mapper.size, global_pattern)
    for ii in range(num_global_subdomains):
        block_space.mapper.copy_local_to_global(local_matrices[ii], local_patterns[ii], ii, system_matrix)
        if ii in grid.boundary_subdomains():
            block_space.mapper.copy_local_to_global(boundary_matrices[ii], boundary_patterns[ii],
                                                    ii, ii, system_matrix)
        for jj in grid.neighboring_subdomains(ii):
            if ii < jj:  # Assemble primally (visit each coupling only once).
                block_space.mapper.copy_local_to_global(coupling_matrices['in_in'][(ii, jj)],
                                                        coupling_patterns['in_in'][(ii, jj)],
                                                        ii, ii, system_matrix)
                block_space.mapper.copy_local_to_global(coupling_matrices['out_out'][(ii, jj)],
                                                        coupling_patterns['out_out'][(ii, jj)],
                                                        jj, jj, system_matrix)
                block_space.mapper.copy_local_to_global(coupling_matrices['in_out'][(ii, jj)],
                                                        coupling_patterns['in_out'][(ii, jj)],
                                                        ii, jj, system_matrix)
                block_space.mapper.copy_local_to_global(coupling_matrices['out_in'][(ii, jj)],
                                                        coupling_patterns['out_in'][(ii, jj)],
                                                        jj, ii, system_matrix)
    logger.debug('discretize lhs global op ...')
    op = DuneXTMatrixOperator(system_matrix, dof_communicator=block_space.dof_communicator, solver_options=solver_options)
    logger.debug('discretize lhs global op done ...')
    mats = np.full((num_global_subdomains, num_global_subdomains), None)
    for ii in range(num_global_subdomains):
        ii_size = block_space.local_space(ii).size()
        for jj in range(ii, num_global_subdomains):
            jj_size = block_space.local_space(jj).size()
            if ii == jj:
                mats[ii, ii] = Matrix(ii_size, ii_size, local_patterns[ii])
            elif (ii, jj) in coupling_matrices['in_out']:
                mats[ii, jj] = Matrix(ii_size, jj_size, coupling_patterns['in_out'][(ii, jj)])
                mats[jj, ii] = Matrix(jj_size, ii_size, coupling_patterns['out_in'][(ii, jj)])


    for ii in range(num_global_subdomains):
        for jj in range(ii, num_global_subdomains):
            if ii == jj:
                mats[ii, ii].axpy(1.,  local_matrices[ii])
                if ii in boundary_matrices:
                    mats[ii, ii].axpy(1.,  boundary_matrices[ii])
            elif (ii, jj) in coupling_matrices['in_out']:
                mats[ii, ii].axpy(1., coupling_matrices['in_in'][(ii, jj)])
                mats[jj, jj].axpy(1., coupling_matrices['out_out'][(ii, jj)])
                mats[ii, jj].axpy(1., coupling_matrices['in_out'][(ii, jj)])
                mats[jj, ii].axpy(1., coupling_matrices['out_in'][(ii, jj)])

    logger.debug('discretize lhs block op ...')
    ops = np.full((num_global_subdomains, num_global_subdomains), None)
    for (ii, jj), mat in np.ndenumerate(mats):
        ops[ii, jj] = DuneXTMatrixOperator(mat, name='local_block_{}-{}'.format(ii,jj),
                                           source_id='domain_{}'.format(jj),
                                           range_id='domain_{}'.format(ii)) if mat else None

    block_op = BlockOperator(ops, dof_communicator=block_space.dof_communicator, name='BlockOp')
    return op, block_op


def discretize_rhs(f_func, grid, block_space, global_operator, block_ops, block_op):
    logger = getLogger('discretizing_rhs')
    logger.debug('...')
    local_subdomains, num_local_subdomains, num_global_subdomains = _get_subdomains(grid)
    local_vectors = [None]*num_global_subdomains
    rhs_vector = Vector(block_space.mapper.size, 0.)
    for ii in range(num_global_subdomains):
        local_vectors[ii] = Vector(block_space.local_space(ii).size())
        l2_functional = make_l2_volume_vector_functional(f_func, local_vectors[ii],
                                                         block_space.local_space(ii), over_integrate=2)
        l2_functional.assemble()
        block_space.mapper.copy_local_to_global(local_vectors[ii], ii, rhs_vector)
    rhs = VectorFunctional(global_operator.range.make_array([rhs_vector]))
    rhss = []
    for ii in range(num_global_subdomains):
        rhss.append(block_ops[0]._blocks[ii, ii].range.make_array([local_vectors[ii]]))
    block_rhs = VectorFunctional(block_op.range.make_array(rhss))
    return rhs, block_rhs


def discretize(grid_and_problem_data, solver_options, mpi_comm):
    ################ Setup

    logger = getLogger('discretize_elliptic_block_swipdg.discretize')
    logger.info('discretizing ... ')

    grid, boundary_info = grid_and_problem_data['grid'], grid_and_problem_data['boundary_info']
    local_all_dirichlet_boundary_info = make_subdomain_boundary_info(
        grid, {'type': 'xt.grid.boundaryinfo.alldirichlet'}
    )
    local_subdomains, num_local_subdomains, num_global_subdomains = _get_subdomains(grid)
    local_all_neumann_boundary_info = make_subdomain_boundary_info(grid, {'type': 'xt.grid.boundaryinfo.allneumann'})

    block_space = make_block_dg_space(grid)
    global_rt_space = make_rt_space(grid)
    subdomain_rt_spaces = [global_rt_space.restrict_to_dd_subdomain_view(grid, ii)
                           for ii in range(num_global_subdomains)]

    local_patterns = [block_space.local_space(ii).compute_pattern('face_and_volume')
                      for ii in range(block_space.num_blocks)]
    coupling_patterns = {'in_in' : {}, 'out_out' : {}, 'in_out' : {}, 'out_in' : {}}
    coupling_matrices = {'in_in': {}, 'out_out': {}, 'in_out': {}, 'out_in': {}}

    for ii in range(num_global_subdomains):
        ii_size = block_space.local_space(ii).size()
        for jj in grid.neighboring_subdomains(ii):
            jj_size = block_space.local_space(jj).size()
            if ii < jj:  # Assemble primally (visit each coupling only once).
                coupling_patterns['in_in'][(ii, jj)] = block_space.local_space(ii).compute_pattern('face_and_volume')
                coupling_patterns['out_out'][(ii, jj)] = block_space.local_space(jj).compute_pattern('face_and_volume')
                coupling_patterns['in_out'][(ii, jj)] = block_space.compute_coupling_pattern(ii, jj, 'face')
                coupling_patterns['out_in'][(ii, jj)] = block_space.compute_coupling_pattern(jj, ii, 'face')
                coupling_matrices['in_in'][(ii, jj)] = Matrix(ii_size, ii_size, coupling_patterns['in_in'][(ii, jj)])
                coupling_matrices['out_out'][(ii, jj)] = Matrix(jj_size, jj_size, coupling_patterns['out_out'][(ii, jj)])
                coupling_matrices['in_out'][(ii, jj)] = Matrix(ii_size, jj_size, coupling_patterns['in_out'][(ii, jj)])
                coupling_matrices['out_in'][(ii, jj)] = Matrix(jj_size, ii_size, coupling_patterns['out_in'][(ii, jj)])
    boundary_patterns = {}
    for ii in grid.boundary_subdomains():
        boundary_patterns[ii] = block_space.local_space(ii).compute_pattern('face_and_volume')

    ################ Assemble LHS and RHS

    lambda_, kappa = grid_and_problem_data['lambda'], grid_and_problem_data['kappa']
    if isinstance(lambda_, dict):
        lambda_funcs = lambda_['functions']
        lambda_coeffs = lambda_['coefficients']
    else:
        lambda_funcs = [lambda_,]
        lambda_coeffs = [1,]

    logger.debug('block op ... ')
    ops, block_ops = zip(*(discretize_lhs(lf, grid, block_space, local_patterns, boundary_patterns,
                                          coupling_matrices, kappa, local_all_neumann_boundary_info, boundary_info,
                                          coupling_patterns, solver_options) for lf in lambda_funcs))
    global_operator = LincombOperator(ops, lambda_coeffs, solver_options=solver_options, name='GlobalOperator')
    logger.debug('block op global done ')
    block_op = LincombOperator(block_ops, lambda_coeffs, name='lhs', solver_options=solver_options)
    logger.debug('block op done ')

    f = grid_and_problem_data['f']
    if isinstance(f, dict):
        f_funcs = f['functions']
        f_coeffs = f['coefficients']
    else:
        f_funcs = [f,]
        f_coeffs = [1,]
    rhss, block_rhss = zip(*(discretize_rhs(ff, grid, block_space, global_operator, block_ops, block_op) for ff in f_funcs))
    global_rhs = LincombOperator(rhss, f_coeffs)
    block_rhs = LincombOperator(block_rhss, f_coeffs)

    solution_space = block_op.source

    ################ Assemble interpolation and reconstruction operators
    logger.info('discretizing interpolation ')

    # Oswald interpolation error operator
    oi_op = BlockDiagonalOperator([OswaldInterpolationErrorOperator(ii, block_op.source, grid, block_space)
                                   for ii in range(num_global_subdomains)],
                                  name='oswald_interpolation_error')

    # Flux reconstruction operator
    fr_op = LincombOperator(
        [BlockDiagonalOperator([FluxReconstructionOperator(ii, block_op.source, grid, block_space, global_rt_space,
                                                           subdomain_rt_spaces, lambda_xi, kappa)
                                for ii in range(num_global_subdomains)])
         for lambda_xi in lambda_funcs],
        lambda_coeffs,
        name='flux_reconstruction'
    )

    ################ Assemble inner products and error estimator operators
    logger.info('discretizing inner products ')

    lambda_bar, lambda_hat = grid_and_problem_data['lambda_bar'], grid_and_problem_data['lambda_hat']
    mu_bar, mu_hat = grid_and_problem_data['mu_bar'], grid_and_problem_data['mu_hat']
    operators = {}
    local_projections = []
    local_rt_projections = []
    local_oi_projections = []
    local_div_ops = []
    local_l2_products = []
    data = dict(grid=grid,
                block_space=block_space,
                local_projections=local_projections,
                local_rt_projections=local_rt_projections,
                local_oi_projections=local_oi_projections,
                local_div_ops=local_div_ops,
                local_l2_products=local_l2_products)

    for ii in range(num_global_subdomains):

        neighborhood = grid.neighborhood_of(ii)
        logger.error('NEIGH {}: {}'.format(ii, neighborhood))

        ################ Assemble local inner products

        local_dg_space = block_space.local_space(ii)
        # we want a larger pattern to allow for axpy with other matrices
        tmp_local_matrix = Matrix(local_dg_space.size(),
                                  local_dg_space.size(),
                                  local_dg_space.compute_pattern('face_and_volume'))
        local_energy_product_ops = []
        local_energy_product_coeffs = []
        for func, coeff in zip(lambda_funcs, lambda_coeffs):
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
        operators[local_energy_product_name] = \
            local_energy_product.assemble(mu_bar).with_(name=local_energy_product_name)

        local_l2_product = DuneXTMatrixOperator(local_l2_product.matrix(),
                                                source_id='domain_{}'.format(ii),
                                                range_id='domain_{}'.format(ii))
        local_l2_products.append(local_l2_product)

        # assemble local elliptic product
        matrix = make_local_elliptic_matrix_operator(grid, ii,
                                                     local_dg_space,
                                                     lambda_bar, kappa)
        matrix.assemble()
        local_elliptic_product = DuneXTMatrixOperator(matrix.matrix(),
                                                      range_id='domain_{}'.format(ii),
                                                      source_id='domain_{}'.format(ii))

        ################ Assemble local to global projections

        # assemble projection (solution space) ->  (ii space)
        local_projection = BlockProjectionOperator(block_op.source, ii)
        local_projections.append(local_projection)

        # assemble projection (RT spaces on neighborhoods of subdomains) ->  (local RT space on ii)
        ops = np.full(num_global_subdomains, None)
        for kk in neighborhood:
            component = grid.neighborhood_of(kk).index(ii)
            assert fr_op.range.subspaces[kk].subspaces[component].id == 'LOCALRT_{}'.format(ii)
            ops[kk] = BlockProjectionOperator(fr_op.range.subspaces[kk], component)
        local_rt_projection = BlockRowOperator(ops, source_spaces=fr_op.range.subspaces,
                                               name='local_rt_projection_{}'.format(ii))
        local_rt_projections.append(local_rt_projection)

        # assemble projection (OI spaces on neighborhoods of subdomains) ->  (ii space)
        ops = np.full(num_global_subdomains, None)
        for kk in neighborhood:
            component = grid.neighborhood_of(kk).index(ii)
            assert oi_op.range.subspaces[kk].subspaces[component].id == 'domain_{}'.format(ii)
            ops[kk] = BlockProjectionOperator(oi_op.range.subspaces[kk], component)
        local_oi_projection = BlockRowOperator(ops, source_spaces=oi_op.range.subspaces,
                                               name='local_oi_projection_{}'.format(ii))
        local_oi_projections.append(local_oi_projection)

        ################ Assemble additional operators for error estimation

        # assemble local divergence operator
        local_rt_space = global_rt_space.restrict_to_dd_subdomain_view(grid, ii)
        local_div_op = make_divergence_matrix_operator_on_subdomain(grid, ii, local_dg_space, local_rt_space)
        local_div_op.assemble()
        local_div_op = DuneXTMatrixOperator(local_div_op.matrix(),
                                            source_id='LOCALRT_{}'.format(ii),
                                            range_id='domain_{}'.format(ii),
                                            name='local_divergence_{}'.format(ii))
        local_div_ops.append(local_div_op)

        ################ Assemble error estimator operators -- Nonconformity

        operators['nc_{}'.format(ii)] = \
            Concatenation([local_oi_projection.T, local_elliptic_product, local_oi_projection],
                          name='nonconformity_{}'.format(ii))

        ################ Assemble error estimator operators -- Residual

        if len(f_funcs) == 1:
            assert f_coeffs[0] == 1
            local_div = Concatenation([local_div_op, local_rt_projection])
            local_rhs = VectorFunctional(block_rhs.operators[0]._array._blocks[ii])

            operators['r_fd_{}'.format(ii)] = \
                Concatenation([local_rhs, local_div], name='r1_{}'.format(ii))

            operators['r_dd_{}'.format(ii)] = \
                Concatenation([local_div.T, local_l2_product, local_div], name='r2_{}'.format(ii))

        ################ Assemble error estimator operators -- Diffusive flux

        operators['df_aa_{}'.format(ii)] = LincombOperator(
            [assemble_estimator_diffusive_flux_aa(lambda_xi, lambda_xi_prime, grid, ii, block_space, lambda_hat, kappa,
                                                  solution_space)
             for lambda_xi in lambda_funcs
             for lambda_xi_prime in lambda_funcs],
            [ProductParameterFunctional([c1, c2])
             for c1 in lambda_coeffs
             for c2 in lambda_coeffs],
            name='diffusive_flux_aa_{}'.format(ii))

        operators['df_bb_{}'.format(ii)] = assemble_estimator_diffusive_flux_bb(grid, ii, subdomain_rt_spaces,
                                                                                lambda_hat, kappa, local_rt_projection)

        operators['df_ab_{}'.format(ii)] = LincombOperator(
            [assemble_estimator_diffusive_flux_ab(lambda_xi, grid, ii, block_space, subdomain_rt_spaces, lambda_hat,
                                                  kappa, local_rt_projection, local_projection) for lambda_xi in lambda_funcs],
            lambda_coeffs,
            name='diffusive_flux_ab_{}'.format(ii)
        )

    ################ Final assembly
    logger.info('final assembly ')

    # instantiate error estimator
    min_diffusion_evs = np.array([min_diffusion_eigenvalue(grid, ii, lambda_hat, kappa) for ii in
                                  range(num_global_subdomains)])

    subdomain_diameters = np.array([subdomain_diameter(grid, ii) for ii in range(num_global_subdomains)])
    if len(f_funcs) == 1:
        assert f_coeffs[0] == 1
        local_eta_rf_squared = np.array([apply_l2_product(grid, ii, f_funcs[0], f_funcs[0], over_integrate=2) for ii in
                                         range(num_global_subdomains)])
    else:
        local_eta_rf_squared = None
    estimator = EllipticEstimator(grid, min_diffusion_evs, subdomain_diameters, local_eta_rf_squared, lambda_coeffs,
                                  mu_bar, mu_hat, fr_op, oswald_interpolation_error=oi_op,
                                  global_dg_space=block_space,
                                  mpi_comm = mpi_comm, global_rt_space=global_rt_space)
    l2_product = BlockDiagonalOperator(local_l2_products)

    # instantiate discretization
    neighborhoods = [grid.neighborhood_of(ii) for ii in range(num_global_subdomains)]
    local_boundary_info = make_subdomain_boundary_info(grid_and_problem_data['grid'],
                                                       {'type': 'xt.grid.boundaryinfo.alldirichlet'})
    d = DuneDiscretization(global_operator=global_operator,
                           global_rhs=global_rhs,
                           neighborhoods=neighborhoods,
                           enrichment_data=(grid, local_boundary_info, lambda_, kappa, f, block_space),
                           operator=block_op,
                           rhs=block_rhs,
                           visualizer=DuneGDTVisualizer(block_space),
                           operators=operators,
                           products={'l2': l2_product},
                           estimator=estimator,
                           data=data)
    parameter_range = grid_and_problem_data['parameter_range']
    logger.info('final assembly B')
    d = d.with_(parameter_space=CubicParameterSpace(d.parameter_type, parameter_range[0], parameter_range[1]))
    logger.info('final assembly C')
    return d, data

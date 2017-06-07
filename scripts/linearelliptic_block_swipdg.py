#!/usr/bin/env python
from itertools import product

import numpy as np

from pymor.basic import *
from pymor.reductors.system import GenericRBSystemReductor


class DuneDiscretization(StationaryDiscretization):

    def _solve(self, mu):
        if not self.logging_disabled:
            self.logger.info('Solving {} for {} ...'.format(self.name, mu))

        return self.solution_space.from_data(
            self.operators['global_op'].apply_inverse(self.operators['global_rhs'].as_vector(mu=mu),
                                                      mu=mu).data
        )

    def as_generic_type(self):
        return StationaryDiscretization(self.operator, self.rhs, parameter_space=self.parameter_space)

    def visualize(self, U, *args, **kwargs):
        U = self.operators['global_op'].source.from_data(U.data)
        self.visualizer.visualize(U, self, *args, **kwargs)


def discretize():
    from dune.xt.common import init_logger, init_mpi
    init_mpi()
    init_logger()

    from dune.xt.functions import (
        make_checkerboard_function_1x1,
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
        make_system_assembler
    )

    grid = make_grid(lower_left=[0, 0], upper_right=[1, 1], num_elements=[4, 4], num_refinements=4,
                     num_partitions=[3, 3])

    all_dirichlet_boundary_info = make_boundary_info(grid, {'type': 'xt.grid.boundaryinfo.alldirichlet'})
    all_neumann_boundary_info = make_subdomain_boundary_info(grid, {'type': 'xt.grid.boundaryinfo.allneumann'})

    XBLOCKS = YBLOCKS = 2

    def diffusion_function_factory(ix, iy):
        values = [[0.]]*(YBLOCKS*XBLOCKS)
        values[ix + XBLOCKS*iy] = [1.]
        return make_checkerboard_function_1x1(grid_provider=grid, lower_left=[0, 0], upper_right=[1, 1],
                                              num_elements=[XBLOCKS, YBLOCKS],
                                              values=values, name='diffusion_{}_{}'.format(ix, iy))

    diffusion_functions = [diffusion_function_factory(ix, iy)
                           for ix, iy in product(range(XBLOCKS), range(YBLOCKS))]

    # values = [[0.1]]*(YBLOCKS*XBLOCKS)
    # values[0] = [1.]
    # lambdas = make_checkerboard_function_1x1(grid_provider=grid, lower_left=[0, 0], upper_right=[1, 1],
    #                                          num_elements=[XBLOCKS, YBLOCKS],
    #                                          values=values, name='diffusion')
    kappa = make_constant_function_2x2(grid, [[1., 0.], [0., 1.]], name='diffusion')
    f = make_expression_function_1x1(grid, 'x', '1', order=0, name='force')

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
        from pymor.operators.block import BlockOperator

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
    block_op = LincombOperator(block_ops, coefficients)

    from pymor.bindings.dunegdt import DuneGDTVisualizer

    d = DuneDiscretization(block_op, block_rhs, visualizer=DuneGDTVisualizer(block_space),
                           operators={'global_op': op, 'global_rhs': rhs})
    d = d.with_(parameter_space=CubicParameterSpace(d.parameter_type, 0.1, 1.))

    return d


d = discretize()


# print(d.operators['global_op'].matrix.sup_norm())

U = d.solution_space.empty()
for mu in d.parameter_space.sample_uniformly(2):
    U.append(d.solve(mu))
d.visualize(U, filename='U')
bases = {b.space.id: b.copy() for b in U._blocks}
for V in bases.values():
    gram_schmidt(V, copy=False)
reductor = GenericRBSystemReductor(d, bases)
rd = reductor.reduce()
u = rd.solution_space.empty()
for mu in d.parameter_space.sample_uniformly(2):
    u.append(rd.solve(mu))
UU = reductor.reconstruct(u)
print((U - UU).l2_norm() / U.l2_norm())

# result = reduction_error_analysis(rd, d, reductor, test_mus=10, estimator=False, condition=True, plot=True)
# print(result['summary'])

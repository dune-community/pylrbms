#!/usr/bin/env python
import numpy as np

from pymor.discretizations.basic import StationaryDiscretization


class DuneDiscretization(StationaryDiscretization):

    def _solve(self, mu):
        return self.solution_space.from_data(
            self.operators['global_op'].apply_inverse(self.operators['global_rhs'].as_vector()).data
        )

    def as_generic_type(self):
        return StationaryDiscretization(self.operator, self.rhs)

    def visualize(self, U, *args, **kwargs):
        U = self.operators['global_op'].source.from_data(U.data)
        self.visualizer.visualize(U, self, *args, **kwargs)


def discretize():
    from dune.xt.common import init_logger, init_mpi
    from dune.xt.grid import (
        make_cube_dd_subdomains_grid__2d_simplex_aluconform as make_grid,
        make_boundary_info_on_dd_subdomain_layer as make_boundary_info,
        make_apply_on_inner_intersections_2d_simplex_aluconformgrid_dd_subdomain_coupling_part as make_apply_on_inner_intersections  # NOQA
    )
    from dune.xt.functions import make_constant_function_1x1, make_constant_function_2x2, make_expression_function_1x1
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
        make_system_assembler
    )

    init_mpi()
    init_logger()

    inner_boundary_id = 18446744073709551573
    grid = make_grid(lower_left=[-1, -1], upper_right=[1, 1], num_elements=[4, 4], num_refinements=4,
                     num_partitions=[2, 2], num_oversampling_layers=0, inner_boundary_segment_index=inner_boundary_id)
    grid.visualize('../block_swipdg_esv2007_grid', with_coupling=True)

    boundary_info = make_boundary_info(grid, {'type': 'xt.grid.boundaryinfo.boundarysegmentindexbased',
                                              'default': 'dirichlet',
                                              'neumann': '[{} {}]'.format(inner_boundary_id, inner_boundary_id+1)})

    lmbda = make_constant_function_1x1(grid, 1.0, name='diffusion')
    kappa = make_constant_function_2x2(grid, [[1., 0.], [0., 1.]], name='diffusion')
    f = make_expression_function_1x1(grid, 'x', '0.5*pi*pi*cos(0.5*pi*x[0])*cos(0.5*pi*x[1])', order=3, name='force')

    block_space = make_block_space(grid)

    print('preparing local and coupling containers ...')
    local_patterns = [block_space.local_space(ii).compute_pattern('face_and_volume')
                      for ii in np.arange(block_space.num_blocks)]
    coupling_patterns = {}
    for ii in np.arange(grid.num_subdomains):
        for jj in grid.neighboring_subdomains(ii):
                coupling_patterns[(ii, jj)] = block_space.compute_coupling_pattern(ii, jj, 'face')

    local_matrices = [None]*grid.num_subdomains
    local_vectors = [None]*grid.num_subdomains
    coupling_matrices = {}
    for ii in np.arange(grid.num_subdomains):
        local_matrices[ii] = Matrix(block_space.local_space(ii).size(),
                                    block_space.local_space(ii).size(),
                                    local_patterns[ii])
        local_vectors[ii] = Vector(block_space.local_space(ii).size())
    for ii in np.arange(grid.num_subdomains):
        for jj in grid.neighboring_subdomains(ii):
            coupling_matrices[(ii, jj)] = Matrix(block_space.local_space(ii).size(),
                                                 block_space.local_space(jj).size(),
                                                 coupling_patterns[(ii, jj)])

    print('assembling local containers ...')

    def assemble_local_contributions(subdomain):
        ipdg_operator = make_elliptic_swipdg_matrix_operator(lmbda, kappa, boundary_info, local_matrices[subdomain],
                                                             block_space.local_space(subdomain))
        l2_functional = make_l2_volume_vector_functional(f, local_vectors[subdomain],
                                                         block_space.local_space(subdomain))
        local_assembler = make_system_assembler(block_space.local_space(subdomain))
        local_assembler.append(ipdg_operator)
        local_assembler.append(l2_functional)
        local_assembler.assemble()

    for ii in np.arange(grid.num_subdomains):
        assemble_local_contributions(ii)

    print('assembling coupling matrices ...')

    local_ipdg_coupling_operator = make_local_elliptic_swipdg_coupling_operator(lmbda, kappa)
    apply_on_inner_intersections = make_apply_on_inner_intersections()

    def assemble_coupling_contributions(subdomain, neighboring_subdomain):
        coupling_assembler = block_space.coupling_assembler(subdomain, neighboring_subdomain)
        coupling_assembler.append(local_ipdg_coupling_operator,
                                  local_matrices[subdomain],
                                  local_matrices[neighboring_subdomain],
                                  coupling_matrices[(subdomain, neighboring_subdomain)],
                                  coupling_matrices[(neighboring_subdomain, subdomain)],
                                  apply_on_inner_intersections)

    for ii in np.arange(grid.num_subdomains):
        for jj in grid.neighboring_subdomains(ii):
            if ii < jj:
                assemble_coupling_contributions(ii, jj)

    print('creating global container ...')

    global_pattern = SparsityPatternDefault(block_space.mapper.size)
    for ii in np.arange(grid.num_subdomains):
        block_space.mapper.copy_local_to_global(local_patterns[ii], ii, global_pattern)
        for jj in grid.neighboring_subdomains(ii):
            block_space.mapper.copy_local_to_global(coupling_patterns[(ii, jj)], ii, jj, global_pattern)
            block_space.mapper.copy_local_to_global(coupling_patterns[(jj, ii)], jj, ii, global_pattern)

    system_matrix = Matrix(block_space.mapper.size, block_space.mapper.size, global_pattern)
    rhs_vector = Vector(block_space.mapper.size, 0.)
    for ii in np.arange(grid.num_subdomains):
        block_space.mapper.copy_local_to_global(local_matrices[ii], local_patterns[ii], ii, system_matrix)
        block_space.mapper.copy_local_to_global(local_vectors[ii], ii, rhs_vector)
        for jj in grid.neighboring_subdomains(ii):
            block_space.mapper.copy_local_to_global(coupling_matrices[(ii, jj)],
                                                    coupling_patterns[(ii, jj)],
                                                    ii, jj, system_matrix)
            block_space.mapper.copy_local_to_global(coupling_matrices[(jj, ii)],
                                                    coupling_patterns[(jj, ii)],
                                                    jj, ii, system_matrix)

    from pymor.bindings.dunext import DuneXTMatrixOperator
    from pymor.bindings.dunegdt import DuneGDTVisualizer
    from pymor.operators.constructions import VectorFunctional
    from pymor.operators.block import BlockOperator

    op = DuneXTMatrixOperator(system_matrix)
    ops = np.full((grid.num_subdomains, grid.num_subdomains), None)
    rhss = []
    for ii in range(grid.num_subdomains):
        for jj in range(grid.num_subdomains):
            if ii == jj:
                ops[ii, jj] = o = DuneXTMatrixOperator(local_matrices[ii],
                                                       source_id='domain_{}'.format(jj),
                                                       range_id='domain_{}'.format(ii))
                rhss.append(o.range.make_array([local_vectors[ii]]))
            elif (ii, jj) in coupling_matrices:
                ops[ii, jj] = DuneXTMatrixOperator(coupling_matrices[(ii, jj)],
                                                   source_id='domain_{}'.format(jj),
                                                   range_id='domain_{}'.format(ii))
    block_op = BlockOperator(ops)
    block_rhs = VectorFunctional(block_op.range.make_array(rhss))

    rhs = VectorFunctional(op.range.make_array([rhs_vector]))
    d = DuneDiscretization(block_op, block_rhs, visualizer=DuneGDTVisualizer(block_space),
                           operators={'global_op': op, 'global_rhs': rhs})

    return d

d = discretize()
# d.visualize(d.solve(), filename='foo')

from pymor.reductors.system import GenericRBSystemReductor

U = d.solve()
bases = {b.space.id: b.copy() for b in U._blocks}
reductor = GenericRBSystemReductor(d, bases)
rd = reductor.reduce()
u = rd.solve()
UU = reductor.reconstruct(u)
print((U - UU).l2_norm() / U.l2_norm())

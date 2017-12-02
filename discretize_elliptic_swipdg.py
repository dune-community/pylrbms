import numpy as np

import dune.gdt
from dune.gdt import (
        make_elliptic_matrix_operator_istl_row_major_sparse_matrix_double
          as make_elliptic_matrix_operator,
        make_elliptic_swipdg_affine_factor_matrix_operator_istl_row_major_sparse_matrix_double
          as make_elliptic_swipdg_matrix_operator,
        make_l2_matrix_operator_istl_row_major_sparse_matrix_double
          as make_l2_matrix_operator,
        make_l2_volume_vector_functional_istl_dense_vector_double
          as make_l2_volume_vector_functional,
        make_system_assembler
        )

from pymor.bindings.dunegdt import DuneGDTVisualizer
from pymor.bindings.dunext import DuneXTMatrixOperator, DuneXTVectorSpace
from pymor.core.interfaces import ImmutableInterface
from pymor.core.logger import getLogger
from pymor.discretizations.basic import StationaryDiscretization
from pymor.operators.basic import OperatorBase
from pymor.operators.constructions import LincombOperator, VectorFunctional, Concatenation
from pymor.parameters.functionals import ProductParameterFunctional
from pymor.parameters.spaces import CubicParameterSpace


def discretize(grid_and_problem_data, polorder):

    logger = getLogger('discretize_elliptic_swipdg.discretize')
    logger.info('discretizing ... ')
    over_integrate = 2

    grid, boundary_info = grid_and_problem_data['grid'], grid_and_problem_data['boundary_info']

    affine_lambda, kappa, f = (grid_and_problem_data['lambda'],
                               grid_and_problem_data['kappa'],
                               grid_and_problem_data['f'])
    lambda_bar, lambda_bar = grid_and_problem_data['lambda_bar'], grid_and_problem_data['lambda_bar']
    mu_bar, mu_hat, parameter_range  = (grid_and_problem_data['mu_bar'],
                                        grid_and_problem_data['mu_hat'],
                                        grid_and_problem_data['parameter_range'])

    # create discrete function space
    make_space = 'make_dg_leaf_part_to_1x1_fem_p{}_space'.format(polorder)
    if not make_space in dune.gdt.__dict__:
        raise RuntimeError('Not available for polynomial order {}!'.format(polorder))
    make_space = dune.gdt.__dict__[make_space]
    space = make_space(grid)
    # prepare operators and functionals
    system_ops = [make_elliptic_swipdg_matrix_operator(diffusion_factor, kappa, boundary_info, space, over_integrate)
                  for diffusion_factor in affine_lambda['functions']]
    functional = make_l2_volume_vector_functional(f, space, over_integrate)
    l2_operator = make_l2_matrix_operator(space)
    elliptic_ops = [make_elliptic_matrix_operator(diffusion_factor, kappa, space, over_integrate)
                    for diffusion_factor in affine_lambda['functions']]
    # assemble everything in one grid walk
    system_assembler = make_system_assembler(space)
    for op in system_ops:
        system_assembler.append(op)
    for op in elliptic_ops:
        system_assembler.append(op)
    system_assembler.append(functional)
    system_assembler.append(l2_operator)
    system_assembler.walk()
    # wrap everything
    op = LincombOperator([DuneXTMatrixOperator(o.matrix()) for o in system_ops],
                         affine_lambda['coefficients'])
    elliptic_op = LincombOperator([DuneXTMatrixOperator(o.matrix()) for o in elliptic_ops],
                                  affine_lambda['coefficients'])
    rhs = VectorFunctional(op.range.make_array([functional.vector()]))
    operators = {'l2': DuneXTMatrixOperator(l2_operator.matrix()),
                 'elliptic': elliptic_op,
                 'elliptic_mu_bar': DuneXTMatrixOperator(elliptic_op.assemble(mu=mu_bar).matrix)}
    d = StationaryDiscretization(op, rhs, operators=operators)

    return d, {'space': space}


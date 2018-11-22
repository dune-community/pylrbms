from dune.gdt.spaces import make_dg_space
from dune.gdt import (
        make_elliptic_matrix_operator_istl_row_major_sparse_matrix_double
          as make_elliptic_matrix_operator,
        make_elliptic_swipdg_affine_factor_matrix_operator_istl_row_major_sparse_matrix_double
          as make_elliptic_swipdg_matrix_operator,
        make_l2_matrix_operator,
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


def discretize(grid_and_problem_data, polorder=1, solver_options=None):

    logger = getLogger('discretize_elliptic_swipdg.discretize')
    logger.info('discretizing ... ')
    over_integrate = 2

    grid, boundary_info = grid_and_problem_data['grid'], grid_and_problem_data['boundary_info']

    _lambda, kappa, f = (grid_and_problem_data['lambda'],
                         grid_and_problem_data['kappa'],
                         grid_and_problem_data['f'])
    lambda_bar, lambda_bar = grid_and_problem_data['lambda_bar'], grid_and_problem_data['lambda_bar']
    mu_bar, mu_hat, parameter_range  = (grid_and_problem_data['mu_bar'],
                                        grid_and_problem_data['mu_hat'],
                                        grid_and_problem_data['parameter_range'])
    space = make_dg_space(grid)
    # prepare operators and functionals
    if isinstance(_lambda, dict):
        system_ops = [make_elliptic_swipdg_matrix_operator(lambda_func, kappa, boundary_info, space, over_integrate)
                      for lambda_func in _lambda['functions']]
        elliptic_ops = [make_elliptic_matrix_operator(lambda_func, kappa, space, over_integrate)
                        for lambda_func in _lambda['functions']]
    else:
        system_ops = [make_elliptic_swipdg_matrix_operator(_lambda, kappa, boundary_info, space, over_integrate), ]
        elliptic_ops = [make_elliptic_matrix_operator(_lambda, kappa, space, over_integrate), ]
    if isinstance(f, dict):
        rhs_functionals = [make_l2_volume_vector_functional(f_func, space, over_integrate)
                           for f_func in f['functions']]
    else:
        rhs_functionals = [make_l2_volume_vector_functional(f, space, over_integrate), ]
    l2_matrix_with_system_pattern = system_ops[0].matrix().copy()
    l2_operator = make_l2_matrix_operator(l2_matrix_with_system_pattern, space)
    # assemble everything in one grid walk
    system_assembler = make_system_assembler(space)
    for op in system_ops:
        system_assembler.append(op)
    for op in elliptic_ops:
        system_assembler.append(op)
    for func in rhs_functionals:
        system_assembler.append(func)
    system_assembler.append(l2_operator)
    system_assembler.walk()
    # wrap everything
    if isinstance(_lambda, dict):
        op = LincombOperator([DuneXTMatrixOperator(o.matrix(), dof_communicator=space.dof_communicator) for o in system_ops],
                             _lambda['coefficients'])
        elliptic_op = LincombOperator([DuneXTMatrixOperator(o.matrix()) for o in elliptic_ops],
                                      _lambda['coefficients'])
    else:
        op = DuneXTMatrixOperator(system_ops[0].matrix())
        elliptic_op = DuneXTMatrixOperator(elliptic_ops[0].matrix())
    if isinstance(f, dict):
        rhs = LincombOperator([VectorFunctional(op.range.make_array([func.vector()]))
                               for func in rhs_functionals],
                              f['coefficients'])
    else:
        rhs = VectorFunctional(op.range.make_array([rhs_functionals[0].vector()]))
    operators = {'l2': DuneXTMatrixOperator(l2_matrix_with_system_pattern),
                 'elliptic': elliptic_op,
                 'elliptic_mu_bar': DuneXTMatrixOperator(elliptic_op.assemble(mu=mu_bar).matrix)}
    d = StationaryDiscretization(op, rhs, operators=operators, visualizer=DuneGDTVisualizer(space))
    d = d.with_(parameter_space=CubicParameterSpace(d.parameter_type, parameter_range[0], parameter_range[1]))

    return d, {'space': space}


#!/usr/bin/env python

from itertools import product

from dune.xt.common import init_logger, init_mpi
try:
    init_mpi()
except:
    pass
# init_logger()


from dune.xt.functions import (
    make_checkerboard_function_1x1,
    make_constant_function_1x1,
    make_constant_function_2x2,
    make_expression_function_1x1
)
from dune.xt.grid import (
    make_boundary_info_on_dd_subdomain_boundary_layer as make_boundary_info,
    make_cube_dd_subdomains_grid__2d_simplex_aluconform as make_grid,
)

from pymor.core.logger import getLogger
from pymor.parameters.functionals import ExpressionParameterFunctional


def init_grid_and_problem(config):
    logger = getLogger('OS2015_academic_problem.OS2015_academic_problem')
    logger.info('initializing grid and problem ... ')

    lower_left = [-1, -1]
    upper_right = [1, 1]
    inner_boundary_id = 18446744073709551573
    grid = make_grid(lower_left=lower_left,
                     upper_right=upper_right,
                     num_elements=config['num_coarse_grid_elements'],
                     num_refinements=config['num_grid_refinements'],
                     num_partitions=config['num_grid_subdomains'],
                     num_oversampling_layers=config['num_grid_oversampling_layers'],
                     inner_boundary_segment_index=inner_boundary_id)

    all_dirichlet_boundary_info = make_boundary_info(grid, {'type': 'xt.grid.boundaryinfo.alldirichlet'})

    diffusion_functions = [make_expression_function_1x1(
        grid, 'x', '1+(cos(0.5*pi*x[0])*cos(0.5*pi*x[1]))', order=2, name='lambda_0'), ]
    diffusion_functions.append(make_expression_function_1x1(
        grid, 'x', '-1*(cos(0.5*pi*x[0])*cos(0.5*pi*x[1]))', order=2, name='lambda_1'))

    parameter_type = {'diffusion': (1,)}
    coefficients = [ExpressionParameterFunctional('1.', parameter_type),
                    ExpressionParameterFunctional('diffusion', parameter_type)]

    kappa = make_constant_function_2x2(grid, [[1., 0.], [0., 1.]], name='kappa')
    f = make_expression_function_1x1(grid, 'x', '0.5*pi*pi*cos(0.5*pi*x[0])*cos(0.5*pi*x[1])', order=2, name='f')
    lambda_bar = make_constant_function_1x1(grid, 1., name='lambda_bar')
    lambda_hat = make_constant_function_1x1(grid, 1., name='lambda_hat')

    return {'grid': grid,
            'boundary_info': all_dirichlet_boundary_info,
            'inner_boundary_id': inner_boundary_id,
            'lambda': {'functions': diffusion_functions,
                       'coefficients': coefficients},
            'lambda_bar': lambda_bar,
            'lambda_hat': lambda_hat,
            'kappa': kappa,
            'f': f,
            'parameter_type': parameter_type,
            'mu_bar': (1,),
            'mu_hat': (1,),
            'mu_min': (0.1,),
            'mu_max': (1,),
            'parameter_range': (0.1, 1)}


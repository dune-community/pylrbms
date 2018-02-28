#!/usr/bin/env python

from itertools import product

import numpy as np

from dune.xt.functions import (
    make_checkerboard_function_1x1,
    make_constant_function_1x1,
    make_constant_function_2x2,
    make_expression_function_1x1
)
from dune.xt.grid import (
    make_boundary_info_on_dd_subdomain_boundary_layer as make_boundary_info
)

from pymor.core.logger import getLogger
from pymor.parameters.functionals import ExpressionParameterFunctional

from grid import make_grid


def init_grid_and_problem(config):
    logger = getLogger('local_thermalblock_problem.local_thermalblock_problem')
    logger.info('initializing grid and problem ... ')

    lower_left = [-1, -1]
    upper_right = [1, 1]
    inner_boundary_id = 18446744073709551573
    grid = make_grid((lower_left, upper_right),
                     config['num_subdomains'],
                     config['half_num_fine_elements_per_subdomain_and_dim'],
                     inner_boundary_id)
    all_dirichlet_boundary_info = make_boundary_info(grid, {'type': 'xt.grid.boundaryinfo.alldirichlet'})

    def make_values(background, foreground):
        checkerboard_values = [[background]]*36
        for ii in (7, 25):
            checkerboard_values[ii] = [foreground]
        return checkerboard_values

    diffusion_functions = [make_checkerboard_function_1x1(grid, lower_left, upper_right,
                                                          [6, 6], make_values(1., 0.),
                                                          name='lambda_0'),
                           make_checkerboard_function_1x1(grid, lower_left, upper_right,
                                                          [6, 6], make_values(0., 1.),
                                                          name='lambda_1')]

    parameter_type = {'diffusion': (1,)}
    coefficients = [ExpressionParameterFunctional('1.', parameter_type),
                    ExpressionParameterFunctional('1.1 + sin(diffusion)', parameter_type)]

    kappa = make_constant_function_2x2(grid, [[1., 0.], [0., 1.]], name='kappa')
    f = make_expression_function_1x1(grid, 'x', '0.5*pi*pi*cos(0.5*pi*x[0])*cos(0.5*pi*x[1])', order=2, name='f')
    lambda_bar = make_checkerboard_function_1x1(grid, lower_left, upper_right, [6, 6], make_values(1., 1.1), name='lambda_bar')
    lambda_hat = make_checkerboard_function_1x1(grid, lower_left, upper_right, [6, 6], make_values(1., 1.1), name='lambda_hat')

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
            'mu_bar': (0,),
            'mu_hat': (0,),
            'mu_min': (0,),
            'mu_max': (np.pi,),
            'parameter_range': (0, np.pi)}


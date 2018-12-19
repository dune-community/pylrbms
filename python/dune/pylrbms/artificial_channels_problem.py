#!/usr/bin/env python

from itertools import product

from dune.xt.functions import (
    make_constant_function_1x1,
    make_constant_function_2x2,
    make_indicator_function_1x1
)
from dune.xt.grid.boundaryinfo import (
    make_boundary_info_on_dd_subdomain_boundary_layer as make_boundary_info
)

from pymor.core.logger import getLogger
from pymor.parameters.functionals import ExpressionParameterFunctional, ProjectionParameterFunctional

from dune.pylrbms.grid import make_grid


def init_grid_and_problem(config, mu_bar=(1,), mu_hat=(1,)):
    logger = getLogger('artificial_channels_problem.artificial_channels_problem')
    logger.info('initializing grid and problem ... ')

    lower_left = [0, 0]
    upper_right = [1, 1]

    mu_min = min((0.01,) + mu_bar + mu_hat)
    mu_max = max((1,) + mu_bar + mu_hat)

    inner_boundary_id = 18446744073709551573
    grid = make_grid((lower_left, upper_right),
                     config['num_subdomains'],
                     config['half_num_fine_elements_per_subdomain_and_dim'],
                     inner_boundary_id)

    all_dirichlet_boundary_info = make_boundary_info(grid, {'type': 'xt.grid.boundaryinfo.alldirichlet'})

    def horizontal_channels(value):
        return [[[[1/16, 1/8 - 1/32], [1 - 1/16, 1/8 + 1/32]], value],
                [[[1/16, 3/8 - 1/32], [1 - 1/16, 3/8 + 1/32]], value],
                [[[1/16, 5/8 - 1/32], [1 - 1/16, 5/8 + 1/32]], value],
                [[[1/16, 7/8 - 1/32], [1 - 1/16, 7/8 + 1/32]], value]]

    def fixed_vertical_connections(value):
        return [[[[1/16, 1/8 + 1/32], [1/4 - 1/16, 3/8 - 1/32]], value],
                [[[1/16, 5/8 + 1/32], [1/4 - 1/16, 7/8 - 1/32]], value],
                [[[3/4 + 1/16, 1/8 + 1/32], [1 - 1/16, 3/8 - 1/32]], value],
                [[[3/4 + 1/16, 5/8 + 1/32], [1 - 1/16, 7/8 - 1/32]], value]]

    def switched_vertical_connections(value):
        return [[[[1/16, 3/8 + 1/32], [1/4 - 1/16, 5/8 - 1/32]], value],
                [[[3/4 + 1/16, 3/8 + 1/32], [1 - 1/16, 5/8 - 1/32]], value]]

    diffusion_horizontal_channels = make_indicator_function_1x1(
            grid,
            horizontal_channels(1),
            'horizontal_channels')
    diffusion_fixed_vertical_connections = make_indicator_function_1x1(
            grid,
            fixed_vertical_connections(1),
            'fixed_vertical_connections')
    diffusion_switched_vertical_connections_right = make_indicator_function_1x1(
            grid,
            switched_vertical_connections(1),
            'switched_vertical_connections')
    diffusion_background = (
              make_constant_function_1x1(grid, 1)
            - diffusion_horizontal_channels
            - diffusion_fixed_vertical_connections
            - diffusion_switched_vertical_connections_right)

    parameter_type = {'switch': (1,)}
    lambda_functions = [
            diffusion_background,
            diffusion_horizontal_channels,
            diffusion_fixed_vertical_connections,
            diffusion_switched_vertical_connections_right]
    lambda_coefficients = [
            ExpressionParameterFunctional(str(mu_min), parameter_type),
            ExpressionParameterFunctional(str(mu_max), parameter_type),
            ExpressionParameterFunctional(str(mu_max), parameter_type),
            ProjectionParameterFunctional(component_name='switch',
                                          component_shape=(1,),
                                          coordinates=(0,))]
    kappa = make_constant_function_2x2(grid, [[1., 0.], [0., 1.]], name='kappa')
    f_functions = [
            make_indicator_function_1x1(
                grid,
                [[[[1/16, 5/8 + 1/32], [1/4 - 1/16, 7/8 - 1/32]], 1],],
                'top_left'),
            make_indicator_function_1x1(
                grid,
                [[[[3/4 + 1/16, 1/8 + 1/32], [1 - 1/16, 3/8 - 1/32]], 1],
                 [[[3/4 + 1/16, 5/8 + 1/32], [1 - 1/16, 7/8 - 1/32]], 1]],
                'right')]
    f_coefficients = [
        ExpressionParameterFunctional('sin(2 * 2 * pi * _t) > 0', {'_t': ()}),
        ExpressionParameterFunctional('-1', None)]

    def create_lambda(mu):
        return (  make_constant_function_1x1(grid, mu_min)
                - make_indicator_function_1x1(grid, horizontal_channels(mu_min))
                - make_indicator_function_1x1(grid, fixed_vertical_connections(mu_min))
                - make_indicator_function_1x1(grid, switched_vertical_connections(mu_min))
                + make_indicator_function_1x1(grid, horizontal_channels(mu_max))
                + make_indicator_function_1x1(grid, fixed_vertical_connections(mu_max))
                + make_indicator_function_1x1(grid, switched_vertical_connections(mu[0])))

    return {'grid': grid,
            'boundary_info': all_dirichlet_boundary_info,
            'inner_boundary_id': inner_boundary_id,
            'lambda': {'functions': lambda_functions,
                       'coefficients': lambda_coefficients},
            'lambda_bar': create_lambda(mu_bar),
            'lambda_hat': create_lambda(mu_hat),
            'kappa': kappa,
            'f': {'functions': f_functions,
                  'coefficients': f_coefficients},
            'parameter_type': parameter_type,
            'mu_bar': mu_bar,
            'mu_hat': mu_hat,
            'mu_min': (mu_min,),
            'mu_max': (mu_max,),
            'parameter_range': (mu_min, mu_max)}


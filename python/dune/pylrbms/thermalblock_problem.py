#!/usr/bin/env python

from itertools import product

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
from pymor.parameters.functionals import ProjectionParameterFunctional

from grid import make_grid


def init_grid_and_problem(config, mu_bar=(1, 1, 1, 1), mu_hat=(1, 1, 1, 1)):
    logger = getLogger('thermalblock_problem.thermalblock_problem')
    logger.info('initializing grid and problem ... ')

    lower_left = [-1, -1]
    upper_right = [1, 1]
    inner_boundary_id = 18446744073709551573
    grid = make_grid((lower_left, upper_right),
                     config['num_subdomains'],
                     config['half_num_fine_elements_per_subdomain_and_dim'],
                     inner_boundary_id)
    all_dirichlet_boundary_info = make_boundary_info(grid, {'type': 'xt.grid.boundaryinfo.alldirichlet'})

    XBLOCKS = 2; YBLOCKS = 2

    def diffusion_function_factory(ix, iy):
        values = [[0.]]*(YBLOCKS*XBLOCKS)
        values[ix + XBLOCKS*iy] = [1.]
        return make_checkerboard_function_1x1(grid_provider=grid, lower_left=lower_left, upper_right=upper_right,
                                              num_elements=[XBLOCKS, YBLOCKS],
                                              values=values, name='diffusion_{}_{}'.format(ix, iy))

    diffusion_functions = [diffusion_function_factory(ix, iy)
                           for ix, iy in product(range(XBLOCKS), range(YBLOCKS))]

    parameter_type = {'diffusion': (YBLOCKS, XBLOCKS)}
    coefficients = [ProjectionParameterFunctional(component_name='diffusion',
                                                  component_shape=(YBLOCKS, XBLOCKS),
                                                  coordinates=(YBLOCKS - y - 1, x))
                    for x in range(XBLOCKS) for y in range(YBLOCKS)]

    kappa = make_constant_function_2x2(grid, [[1., 0.], [0., 1.]], name='kappa')
    f = make_expression_function_1x1(grid, 'x', '0.5*pi*pi*cos(0.5*pi*x[0])*cos(0.5*pi*x[1])', order=2, name='f')
    lambda_bar_values = [[0.]]*(YBLOCKS*XBLOCKS)
    lambda_hat_values = [[0.]]*(YBLOCKS*XBLOCKS)
    counter = 0
    for ix in range(YBLOCKS):
        for iy in range(XBLOCKS):
            lambda_bar_values[ix + XBLOCKS*iy] = [coefficients[counter].evaluate(mu_bar)]
            lambda_hat_values[ix + XBLOCKS*iy] = [coefficients[counter].evaluate(mu_hat)]
            counter += 1
    lambda_bar = make_checkerboard_function_1x1(grid_provider=grid, lower_left=lower_left, upper_right=upper_right,
                                                num_elements=[XBLOCKS, YBLOCKS],
                                                values=lambda_bar_values, name='lambda_bar')
    lambda_hat = make_checkerboard_function_1x1(grid_provider=grid, lower_left=lower_left, upper_right=upper_right,
                                                num_elements=[XBLOCKS, YBLOCKS],
                                                values=lambda_hat_values, name='lambda_hat')

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
            'mu_bar': mu_bar,
            'mu_hat': mu_hat,
            'mu_min': (min(0.1, b, h) for b, h in zip(mu_bar, mu_hat)),
            'mu_max': (max(1, b, h) for b, h in zip(mu_bar, mu_hat)),
            'parameter_range': (min((0.1,) + mu_bar + mu_hat), max((1,) + mu_bar + mu_hat))}


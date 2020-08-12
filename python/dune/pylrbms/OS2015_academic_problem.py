#!/usr/bin/env python
from mpi4py import MPI
from itertools import product
import dune

from dune.xt.functions import (
    make_checkerboard_function_1x1,
    make_constant_function_1x1,
    make_constant_function_2x2,
    make_expression_function_1x1
)

from pymor.core.logger import getLogger
from pymor.parameters.functionals import ExpressionParameterFunctional

from dune.pylrbms.grid import make_grid, make_boundary_info, grid_info


def init_grid_and_problem(config, mu_bar = 1, mu_hat = 1, mpi_comm = MPI.COMM_WORLD):
    # assert mpi_comm.Get_size() < MPI.COMM_WORLD.Get_size() or mpi_comm.Get_size() == 1
    logger = getLogger('OS2015_academic_problem.OS2015_academic_problem')
    logger.info('initializing grid and problem ... ')

    lower_left = [-1, -1]
    upper_right = [1, 1]
    inner_boundary_id = 18446744073709551573
    grid = make_grid((lower_left, upper_right),
                     config['num_subdomains'],
                     config['half_num_fine_elements_per_subdomain_and_dim'],
                     inner_boundary_id,
                     mpi_comm=mpi_comm)
    grid_info(logger.error, grid, mpi_comm)
    all_dirichlet_boundary_info = make_boundary_info(grid, {'type': 'xt.grid.boundaryinfo.alldirichlet'})

    cos = '(cos(0.5*pi*x[0])*cos(0.5*pi*x[1]))'
    diffusion_functions = [make_expression_function_1x1(
        grid, 'x', '1+{}'.format(cos), order=2, name='lambda_0'),
        make_expression_function_1x1(grid, 'x', '-1*{}'.format(cos), order=2, name='lambda_1')]
    # diffusion_functions = [make_expression_function_1x1(
    #     grid, 'x', '1', order=2, name='lambda_0'),
    #     make_expression_function_1x1(grid, 'x', 'x[0]', order=2, name='lambda_1')]
    parameter_type = {'diffusion': (1,)}
    coefficients = [ExpressionParameterFunctional('1.', parameter_type),
                    ExpressionParameterFunctional('diffusion', parameter_type)]

    kappa = make_constant_function_2x2(grid, [[1., 0.], [0., 1.]], name='kappa')
    f = make_expression_function_1x1(grid, 'x', '0.5*pi*pi*{}'.format(cos), order=2, name='f')
    mbc = '1+(1-{})*{}'.format(mu_bar, cos)
    lambda_bar = make_expression_function_1x1(grid, 'x', mbc, order=2, name='lambda_bar')
    lambda_hat = make_expression_function_1x1(grid, 'x', mbc, order=2, name='lambda_hat')

    return {'grid': grid,
            'mpi_comm': mpi_comm,
            'boundary_info': all_dirichlet_boundary_info,
            'inner_boundary_id': inner_boundary_id,
            'lambda': {'functions': diffusion_functions,
                       'coefficients': coefficients},
            'lambda_bar': lambda_bar,
            'lambda_hat': lambda_hat,
            'kappa': kappa,
            'f': f,
            'parameter_type': parameter_type,
            'mu_bar': (mu_bar,),
            'mu_hat': (mu_hat,),
            'mu_min': (min(0.1, mu_bar, mu_hat),),
            'mu_max': (max(1, mu_bar, mu_hat),),
            'parameter_range': (min(0.1, mu_bar, mu_hat), max(1, mu_bar, mu_hat))}


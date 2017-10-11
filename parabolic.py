#!/usr/bin/env python

import numpy as np


from thermalblock_problem import init_grid_and_problem
from discretize_parabolic import discretize

from pymor.core.logger import set_log_levels
set_log_levels({'discretize_elliptic': 'INFO'})


config = {'num_coarse_grid_elements': [4, 4],
          'num_grid_refinements': 2,
          'num_grid_subdomains': [4, 4],
          'num_grid_oversampling_layers': 2} # num_grid_oversampling_layers has to exactly cover one subdomain!


grid_and_problem_data = init_grid_and_problem(config)
grid = grid_and_problem_data['grid']

d, block_space = discretize(grid_and_problem_data, 1., 10)

mu = d.parse_parameter([1, 1., 1., 1.])

U = d.solve(mu)
d.visualize(U, filename='solution')


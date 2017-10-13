#!/usr/bin/env python

import numpy as np

from pymor.core.exceptions import ExtensionError

from thermalblock_problem import init_grid_and_problem
from discretize_parabolic import discretize
from lrbms import LRBMSReductor

from pymor.core.logger import set_log_levels
set_log_levels({'discretize_elliptic': 'INFO',
                'lrbms': 'INFO'})


config = {'num_coarse_grid_elements': [4, 4],
          'num_grid_refinements': 2,
          'num_grid_subdomains': [4, 4],
          'num_grid_oversampling_layers': 2} # num_grid_oversampling_layers has to exactly cover one subdomain!


grid_and_problem_data = init_grid_and_problem(config)
grid = grid_and_problem_data['grid']

d, block_space = discretize(grid_and_problem_data, 1., 10)

# mu = d.parse_parameter([1, 1., 1., 1.])

# for i, mu in enumerate(d.parameter_space.sample_randomly(10)):
#     U = d.solve(mu)
#     d.visualize(U, filename='solution_{}'.format(i))


reductor = LRBMSReductor(d, products=[d.operators['local_energy_dg_product_{}'.format(ii)]
                                    for ii in range(block_space.num_blocks)])

U = d.solution_space.empty()
for mu in d.parameter_space.sample_uniformly(5):
    snapshot = d.solve(mu)
    U.append(snapshot)
    try:
        reductor.extend_basis(snapshot)
    except ExtensionError:
        pass
rd = reductor.reduce()

u = rd.solution_space.empty()
for mu in d.parameter_space.sample_uniformly(5):
    u.append(rd.solve(mu))
UU = reductor.reconstruct(u)
d.visualize(U, filename='full')
d.visualize(UU, filename='red')
d.visualize(U- UU, filename='error')

B = d.solution_space.empty()
for i in range(rd.solution_space.dim):
    u = np.zeros(rd.solution_space.dim)
    u[i] = 1.
    u = rd.solution_space.from_data(u)
    B.append(reductor.reconstruct(u))
d.visualize(B, filename='basis')



print((U - UU).l2_norm() / U.l2_norm())
print(rd.mass)

# u = rd.solve(mu)

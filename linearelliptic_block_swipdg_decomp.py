#!/usr/bin/env python

import numpy as np

from pymor.core.exceptions import ExtensionError

# from thermalblock_problem import init_grid_and_problem
from OS2015_academic_problem import init_grid_and_problem
from discretize_elliptic import discretize
from lrbms import LRBMSReductor

from pymor.core.logger import set_log_levels
set_log_levels({'discretize_elliptic': 'INFO',
                'lrbms': 'INFO'})


config = {'num_coarse_grid_elements': [4, 4],
          'num_grid_refinements': 6,
          'num_grid_subdomains': [8, 8],
          'num_grid_oversampling_layers': 4} # num_grid_oversampling_layers has to exactly cover one subdomain!


grid_and_problem_data = init_grid_and_problem(config)
grid = grid_and_problem_data['grid']
# grid.visualize('grid', False)

d, block_space = discretize(grid_and_problem_data)

# mu = d.parse_parameter([1, 1., 1., 1.])
mu = d.parse_parameter(1.)

U = d.solve(mu)

print('estimating error ', end='', flush=True)

eta, (local_eta_nc, local_eta_r, local_eta_df), _ = d.estimate(U, mu=mu, decompose=True)
print(*enumerate(local_eta_nc))

print('')
print('  nonconformity indicator:  {} (should be 1.66e-01)'.format(np.linalg.norm(local_eta_nc)))
print('  residual indicator:       {} (should be 2.89e-01)'.format(np.linalg.norm(local_eta_r)))
print('  diffusive flux indicator: {} (should be 3.55e-01)'.format(np.linalg.norm(local_eta_df)))
print('  estimated error:          {}'.format(eta))

reductor = LRBMSReductor(d, products=[d.operators['local_energy_dg_product_{}'.format(ii)]
                                      for ii in range(block_space.num_blocks)])
U = d.solution_space.empty()
for mu in d.parameter_space.sample_uniformly(2)[:5]:
    snapshot = d.solve(mu)
    U.append(snapshot)
    try:
        reductor.extend_basis(snapshot)
    except ExtensionError:
        pass
# d.visualize(U, filename='U')
rd = reductor.reduce()

u = rd.solution_space.empty()
for mu in d.parameter_space.sample_uniformly(2)[:5]:
    u.append(rd.solve(mu))
UU = reductor.reconstruct(u)
print((U - UU).l2_norm() / U.l2_norm())

u = rd.solve(mu)

print('estimating reduced error ', end='', flush=True)

eta, (local_eta_nc, local_eta_r, local_eta_df), _ = rd.estimate(u, mu=mu, decompose=True)

print('')
print('  nonconformity indicator:  {} (should be 1.74598)'.format(np.linalg.norm(local_eta_nc)))
print('  residual indicator:       {} (should be 0.28939)'.format(np.linalg.norm(local_eta_r)))
print('  diffusive flux indicator: {} (should be 0.50916)'.format(np.linalg.norm(local_eta_df)))
print('  estimated error:          {}'.format(eta))

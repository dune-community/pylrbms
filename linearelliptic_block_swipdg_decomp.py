#!/usr/bin/env python

import numpy as np
import time

from pymor.core.exceptions import ExtensionError

# from thermalblock_problem import init_grid_and_problem
from OS2015_academic_problem import init_grid_and_problem
from discretize_elliptic_block_swipdg import discretize
from lrbms import LRBMSReductor

from pymor.core.logger import set_log_levels
set_log_levels({'discretize_elliptic_block_swipdg': 'INFO',
                'lrbms': 'INFO'})


config = {'num_subdomains': [4, 4],
          'half_num_fine_elements_per_subdomain_and_dim': 1}


grid_and_problem_data = init_grid_and_problem(config)
grid = grid_and_problem_data['grid']
# grid.visualize('grid', False)

d, d_data = discretize(grid_and_problem_data)
block_space = d_data['block_space']

# mu = d.parse_parameter([1, 1., 1., 1.])
mu = d.parse_parameter(1.)

U = d.solve(mu)

print('estimating error ', end='', flush=True)

eta, (local_eta_nc, local_eta_r, local_eta_df), _ = d.estimate(U, mu=mu, decompose=True)
# print(*enumerate(local_eta_nc))

print('')
print('  nonconformity indicator:  {} (should be 1.66e-01)'.format(np.linalg.norm(local_eta_nc)))
print('  residual indicator:       {} (should be 1.45e-01)'.format(np.linalg.norm(local_eta_r)))
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

tic = time.time()
u = rd.solve(mu)
print('red solve time: ', time.time() - tic)

print('estimating reduced error ', end='', flush=True)

tic = time.time()
eta, (local_eta_nc, local_eta_r, local_eta_df), _ = rd.estimate(u, mu=mu, decompose=True)
print('red est time: ', time.time() - tic)

print('')
print('  nonconformity indicator:  {} (should be 1.66e-01)'.format(np.linalg.norm(local_eta_nc)))
print('  residual indicator:       {} (should be 1.45e-01)'.format(np.linalg.norm(local_eta_r)))
print('  diffusive flux indicator: {} (should be 3.55e-01)'.format(np.linalg.norm(local_eta_df)))
print('  estimated error:          {}'.format(eta))

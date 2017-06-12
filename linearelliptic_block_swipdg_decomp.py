#!/usr/bin/env python

import numpy as np

from pymor.core.exceptions import ExtensionError
from pymor.reductors.system import GenericRBSystemReductor

from thermalblock_problem import init_grid_and_problem
from discretize_elliptic import discretize


config = {'num_coarse_grid_elements': [4, 4],
          'num_grid_refinements': 2,
          'num_grid_subdomains': [2, 2],
          'num_grid_oversampling_layers': 4} # num_grid_oversampling_layers has to exactly cover one subdomain!


grid_and_problem_data = init_grid_and_problem(config)
grid = grid_and_problem_data['grid']

d, block_space, _ = discretize(grid_and_problem_data)

mu = d.parse_parameter([1, 1., 1., 1.])

U = d.solve(mu)

print('estimating error ', end='', flush=True)

eta, (local_eta_nc, local_eta_r, local_eta_df), _ = d.estimate(U, mu=mu, decompose=True)

print('')
print('  nonconformity indicator:  {} (should be 1.66e-01)'.format(np.linalg.norm(local_eta_nc)))
print('  residual indicator:       {} (should be 2.89e-01)'.format(np.linalg.norm(local_eta_r)))
print('  diffusive flux indicator: {} (should be 3.55e-01)'.format(np.linalg.norm(local_eta_df)))
print('  estimated error:          {}'.format(eta))

U = d.solution_space.empty()
reductor = GenericRBSystemReductor(d)
for mu in d.parameter_space.sample_uniformly(2):
    snapshot = d.solve(mu)
    U.append(snapshot)
    try:
        reductor.extend_basis(snapshot)
    except ExtensionError:
        pass
d.visualize(U, filename='U')
rd = reductor.reduce()

u = rd.solution_space.empty()
for mu in d.parameter_space.sample_uniformly(2):
    u.append(rd.solve(mu))
UU = reductor.reconstruct(u)
print((U - UU).l2_norm() / U.l2_norm())

rd = rd.with_(estimator=d.estimator)
u = rd.solve(mu)

print('estimating reduced error ', end='', flush=True)

eta, (local_eta_nc, local_eta_r, local_eta_df), _ = rd.estimate(u, mu=mu, decompose=True)

print('')
print('  nonconformity indicator:  {} (should be 1.66e-01)'.format(np.linalg.norm(local_eta_nc)))
print('  residual indicator:       {} (should be 2.89e-01)'.format(np.linalg.norm(local_eta_r)))
print('  diffusive flux indicator: {} (should be 3.55e-01)'.format(np.linalg.norm(local_eta_df)))
print('  estimated error:          {}'.format(eta))

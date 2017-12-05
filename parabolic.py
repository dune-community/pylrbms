#!/usr/bin/env python

import numpy as np

from pymor.core.exceptions import ExtensionError

from thermalblock_problem import init_grid_and_problem
# from OS2015_academic_problem import init_grid_and_problem
from discretize_parabolic_block_swipdg import discretize
from lrbms import ParabolicLRBMSReductor

from pymor.core.logger import set_log_levels
set_log_levels({'discretize_elliptic_block_swipdg': 'INFO',
                'lrbms': 'INFO',
                'pymor.algorithms.gram_schmidt': 'WARN'})


config = {'num_coarse_grid_elements': [4, 4],
          'num_grid_refinements': 2,
          'num_grid_subdomains': [4, 4],
          'num_grid_oversampling_layers': 2}  # num_grid_oversampling_layers has to exactly cover one subdomain!


grid_and_problem_data = init_grid_and_problem(config)
grid = grid_and_problem_data['grid']

# for nt in [10, 20, 40, 80]:
#     d, _ = discretize(grid_and_problem_data, 1., nt)
#     mu = d.parameter_space.sample_uniformly(1)[0]
#     U = d.solve(mu)
#     print(d.estimate(U, mu))


d, d_data = discretize(grid_and_problem_data, 1., 10)
block_space = d_data['block_space']


# mu = d.parse_parameter([1, 1., 1., 1.])

# for i, mu in enumerate(d.parameter_space.sample_randomly(10)):
#     U = d.solve(mu)
#     d.visualize(U, filename='solution_{}'.format(i))


reductor = ParabolicLRBMSReductor(d, products=[d.operators['local_energy_dg_product_{}'.format(ii)]
                                               for ii in range(block_space.num_blocks)])


mu = d.parameter_space.sample_randomly(1)[0]
# mu = d.parse_parameter(1.)
U = d.solve(mu)
reductor.extend_basis(U)
rd = reductor.reduce()

u = rd.solve(mu)
UU = reductor.reconstruct(u)
# d.visualize(U, filename='full')
# d.visualize(UU, filename='red')
# d.visualize(U - UU, filename='error')

# B = d.solution_space.empty()
# for i in range(rd.solution_space.dim):
#     u = np.zeros(rd.solution_space.dim)
#     u[i] = 1.
#     u = rd.solution_space.from_data(u)
#     B.append(reductor.reconstruct(u))
# d.visualize(B, filename='basis')


print('Relative model reduction errors:')
print((U - UU).l2_norm() / U.l2_norm())
print()

print('Estimated error FOM:')
est, (local_eta_nc, local_eta_r, local_eta_df, time_resiudal, time_deriv_nc) = d.estimate(U, mu)
print('  total estimate:                    {}'.format(est))
print('  elliptic nonconformity indicator:  {}'.format(np.linalg.norm(local_eta_nc)))
print('  elliptic residual indicator:       {}'.format(np.linalg.norm(local_eta_r)))
print('  elliptic diffusive flux indicator: {}'.format(np.linalg.norm(local_eta_df)))
print('  time stepping residual:            {}'.format(np.linalg.norm(time_resiudal)))
print('  time derivative nonconformity:     {}'.format(np.linalg.norm(time_deriv_nc)))
print()

print('Estimated error ROM:')
est, (local_eta_nc, local_eta_r, local_eta_df, time_resiudal, time_deriv_nc) = rd.estimate(u, mu)
print('  total estimate:                    {}'.format(est))
print('  elliptic nonconformity indicator:  {}'.format(np.linalg.norm(local_eta_nc)))
print('  elliptic residual indicator:       {}'.format(np.linalg.norm(local_eta_r)))
print('  elliptic diffusive flux indicator: {}'.format(np.linalg.norm(local_eta_df)))
print('  time stepping residual:            {}'.format(np.linalg.norm(time_resiudal)))
print('  time derivative nonconformity:     {}'.format(np.linalg.norm(time_deriv_nc)))
# print(rd.mass)

# u = rd.solve(mu)

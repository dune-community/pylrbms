#!/usr/bin/env python


import numpy as np
np.seterr(all='raise')

from pymor.core.exceptions import ExtensionError
from pymor.core.logger import getLogger, set_log_levels
set_log_levels({'online_adaptive_lrbms': 'DEBUG',
                'OS2015_academic_problem': 'INFO',
                'discretize_elliptic': 'INFO',
                'offline': 'INFO',
                'online_enrichment': 'INFO'})
logger = getLogger('online_adaptive_lrbms.online_adaptive_lrbms')
from pymor.discretizations.basic import StationaryDiscretization

from OS2015_academic_problem import init_grid_and_problem
# from local_thermalblock_problem import init_grid_and_problem

from discretize_elliptic import (
    Vector,
    discretize,
    make_discrete_function,
)

from offline import init_local_reduced_bases

from online_enrichment import AdaptiveEnrichment


config = {'num_coarse_grid_elements': [4, 4],
          'num_grid_refinements': 2,
          'num_grid_subdomains': [2, 2],
          'num_grid_oversampling_layers': 4, # num_grid_oversampling_layers has to exactly cover one subdomain!
          'initial_RB_order': 1,
          'enrichment_target_error': 0.3, # ([4, 4], 2, [2, 2]): 0.815510144764 | ([4, 4], 6, [8, 8]): 2.25996532203
          'marking_doerfler_theta': 0.8, # ([4, 4], 6, [4, 4]): 0.160346202936
          'marking_max_age': 2} # ([6, 6], 2, [3, 3], 4): 0.28154229174


grid_and_problem_data = init_grid_and_problem(config)
grid = grid_and_problem_data['grid']
# grid.visualize('local_thermalblock_problem_grid', False)


class FakeEstimator(object):

    def __init__(self, disc, reductor):
        self.disc = disc
        self.reductor = reductor

    def estimate(self, U, mu, discretization, decompose=False):
        return self.disc.estimate(self.reductor.reconstruct(U), mu=mu, decompose=decompose)


d, block_space, _ = discretize(grid_and_problem_data)
d.disable_logging()

# logger.info('estimating some errors:')
# errors = []
# for mu in np.linspace(grid_and_problem_data['parameter_range'][0],
#                       grid_and_problem_data['parameter_range'][1],
#                       3):
#     mu = d.parse_parameter(mu)
#     print('  {}: '.format(mu), end='', flush=True)
#     U = d.solve(mu)
#     estimate = d.estimate(U, mu=mu)
#     print(estimate)
#     errors.append(estimate)
# logger.info('')


stripped_d = d.with_(operators={name: op
                                for name, op in d.operators.items() if (
                                    name != 'operator'
                                    and name != 'rhs'
                                    and name[:3] != 'nc_'
                                    and name[:1] != 'r'
                                    and name[:3] != 'df_'
                                    and name[:7] != 'global_')})

reductor = init_local_reduced_bases(grid, stripped_d, block_space, config['initial_RB_order'])

estimator = FakeEstimator(d, reductor)
stripped_d = stripped_d.with_(estimator=estimator)

# logger.info('adding some global solution snapshots to reduced basis ...')
# for mu in (grid_and_problem_data['mu_min'], grid_and_problem_data['mu_max']):
#     U = d.solve(mu)
#     try:
#         reductor.extend_basis(U)
#     except ExtensionError:
#         pass
logger.info('')

with logger.block('reducing ...') as _:
    rd = reductor.reduce()
    rd = rd.with_(estimator=stripped_d.estimator)
logger.info('')

with logger.block('estimating some reduced errors:') as _:
    reduced_errors = []
    for mu in (grid_and_problem_data['mu_min'], grid_and_problem_data['mu_max']):
        mu = d.parse_parameter(mu)
        logger.info('{} ... '.format(mu))
        U = rd.solve(mu)
        estimate = rd.estimate(U, mu=mu)
        logger.info('    {}'.format(estimate))
        reduced_errors.append(estimate)
logger.info('')

logger.info('online phase:')
online_adaptive_LRBMS = AdaptiveEnrichment(grid_and_problem_data, stripped_d, block_space,
                                           reductor, rd, config['enrichment_target_error'],
                                           config['marking_doerfler_theta'],
                                           config['marking_max_age'], fake_estimator=estimator)
for mu in rd.parameter_space.sample_randomly(20):
    U, _, _ = online_adaptive_LRBMS.solve(mu)

logger.info('')
logger.info('finished')


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

from OS2015_academic_problem import (
    make_expression_function_1x1,
    init_grid_and_problem,
)
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
          'initial_RB_order': 0,
          'enrichment_target_error': 1., # ([4, 4], 2, [2, 2]): 0.815510144764 | ([4, 4], 6, [8, 8]): 2.25996532203
          'marking_doerfler_theta': 0.33, # ([4, 4], 6, [4, 4]): 0.160346202936
          'marking_max_age': 2}


grid_and_problem_data = init_grid_and_problem(config)
grid = grid_and_problem_data['grid']
# grid.visualize('grid', False)

d, block_space, local_boundary_info = discretize(grid_and_problem_data)

# logger.info('estimating some errors:')
# errors = []
# for mu in (grid_and_problem_data['mu_min'], grid_and_problem_data['mu_max']):
#     mu = d.parse_parameter(mu)
#     logger.info('  {}: '.format(mu), end='', flush=True)
#     U = d.solve(mu)
#     estimate = d.estimate(U, mu=mu)
#     logger.info(estimate)
#     errors.append(estimate)

# logger.info('')

reductor = init_local_reduced_bases(d, block_space, config['initial_RB_order'])

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
    rd = rd.with_(estimator=d.estimator)
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
online_adaptive_LRBMS = AdaptiveEnrichment(grid_and_problem_data, d, block_space, local_boundary_info,
                                           reductor, rd, config['enrichment_target_error'],
                                           config['marking_doerfler_theta'],
                                           config['marking_max_age'])
for mu in rd.parameter_space.sample_randomly(20):
    U, _, _ = online_adaptive_LRBMS.solve(mu)

logger.info('')
logger.info('finished')


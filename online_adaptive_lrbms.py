#!/usr/bin/env python


import numpy as np
np.seterr(all='raise')

from pymor.core.exceptions import ExtensionError
from pymor.core.logger import getLogger, set_log_levels
set_log_levels({'online_adaptive_lrbms': 'DEBUG',
                'OS2015_academic_problem': 'INFO',
                'discretize_elliptic': 'INFO',
                'offline': 'INFO',
                'online_enrichment': 'INFO',
                'lrbms': 'INFO'})
logger = getLogger('online_adaptive_lrbms.online_adaptive_lrbms')
from pymor.discretizations.basic import StationaryDiscretization

from OS2015_academic_problem import init_grid_and_problem
# from local_thermalblock_problem import init_grid_and_problem
from discretize_elliptic import discretize
from online_enrichment import AdaptiveEnrichment
from lrbms import LRBMSReductor

# max discretization error, to derive enrichment_target_error
# ===========================================================
# OS2015_academic_problem
# [4, 4], 2, [2, 2], 4: 0.815510144764
# [6, 6], 4, [6, 6], 4: 3.03372753518

# local_thermalblock_problem
# [6, 6], 4, [6, 6], 4: 0.585792065793
# ===========================================================

config = {'num_coarse_grid_elements': [4, 4],
          'num_grid_refinements': 2,
          'num_grid_subdomains': [2, 2],
          'num_grid_oversampling_layers': 4, # num_grid_oversampling_layers has to exactly cover one subdomain!
          'initial_RB_order': 0,
          'enrichment_target_error': 1.,
          'marking_doerfler_theta': 0.8,
          'marking_max_age': 2}


grid_and_problem_data = init_grid_and_problem(config)
grid = grid_and_problem_data['grid']
# grid.visualize('grid', False)

d, block_space = discretize(grid_and_problem_data)
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


# The estimator: either we
#  (i)  use the offline/online decomposable estimator (large offline computational effort, instant online estimation); or we
#  (ii) use the high-dimensional estimator (no offline effort, medium online effort).

LRBMS_d = d
reductor = LRBMSReductor(
    d,
    products=[d.operators['local_energy_dg_product_{}'.format(ii)] for ii in range(block_space.num_blocks)],
    order=config['initial_RB_order']
)


# logger.info('adding some global solution snapshots to reduced basis ...')
# for mu in (grid_and_problem_data['mu_min'], grid_and_problem_data['mu_max']):
#     U = LRBMS_d.solve(mu)
#     try:
#         reductor.extend_basis(U)
#     except ExtensionError:
#         pass
# logger.info('')


with logger.block('reducing ...') as _:
    rd = reductor.reduce()
logger.info('')

with logger.block('estimating some reduced errors:') as _:
    for mu in (grid_and_problem_data['mu_min'], grid_and_problem_data['mu_max']):
        mu = rd.parse_parameter(mu)
        logger.info('{} ... '.format(mu))
        U = rd.solve(mu)
        estimate = rd.estimate(U, mu=mu)
        logger.info('    {}'.format(estimate))
logger.info('')

logger.info('online phase:')
online_adaptive_LRBMS = AdaptiveEnrichment(grid_and_problem_data, LRBMS_d, block_space,
                                           reductor, rd, config['enrichment_target_error'],
                                           config['marking_doerfler_theta'],
                                           config['marking_max_age'])
for mu in rd.parameter_space.sample_randomly(20):
    U, _, _ = online_adaptive_LRBMS.solve(mu)

logger.info('')
logger.info('local basis sizes:')
for name, basis in online_adaptive_LRBMS.reductor.bases.items():
    logger.info('{}: {}'.format(name, len(basis)))
logger.info('finished')

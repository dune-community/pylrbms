#!/usr/bin/env python
from mpi4py import MPI
import sys
import pprint
import numpy as np
np.seterr(all='raise')

from pymor.core.exceptions import ExtensionError
from pymor.core.logger import getLogger, set_log_levels
dbg_levels = ({'online_adaptive_lrbms': 'DEBUG',
                'OS2015_academic_problem': 'INFO',
                'discretize_elliptic_block_swipdg': 'DEBUG',
                'offline': 'INFO',
                'pymor.operators.constructions': 'DEBUG',
                'dune.pylrbms.discretize_elliptic_block_swipdg': 'DEBUG',
                'dune.pylrbms': 'DEBUG',
                'pymor.bindings.dunext': 'ERROR',
                'online_enrichment': 'DEBUG',
                'lrbms': 'DEBUG',
                'DXTC': 63})
prod_levels = ({'online_adaptive_lrbms': 'INFO',
                'OS2015_academic_problem': 'INFO',
                'discretize_elliptic_block_swipdg': 'INFO',
                'offline': 'INFO',
                'pymor.operators.constructions': 'ERROR',
                'dune.pylrbms.discretize_elliptic_block_swipdg': 'INFO',
                'dune.pylrbms': 'ERROR',
                'pymor.bindings.dunext': 'ERROR',
                'online_enrichment': 'INFO',
                'lrbms': 'INFO',
                'DXTC': 54})
log_levels = dbg_levels #prod_levels
set_log_levels(log_levels)
logger = getLogger('online_adaptive_lrbms.online_adaptive_lrbms')
from dune.xt.common import logging
logging.create(log_levels['DXTC'])
from pymor.discretizations.basic import StationaryDiscretization

from dune.pylrbms.OS2015_academic_problem import init_grid_and_problem
# from local_thermalblock_problem import init_grid_and_problem
from dune.pylrbms.discretize_elliptic_block_swipdg import discretize
from dune.pylrbms.online_enrichment import AdaptiveEnrichment
from dune.pylrbms.lrbms import LRBMSReductor

# max discretization error, to derive enrichment_target_error
# ===========================================================
# OS2015_academic_problem
# [4, 4], 2, [2, 2], 4: 0.815510144764
# [6, 6], 4, [6, 6], 4: 3.03372753518

# local_thermalblock_problem
# [6, 6], 4, [6, 6], 4: 0.585792065793
# ===========================================================

config = {'num_subdomains': [2, 2],
          'half_num_fine_elements_per_subdomain_and_dim': 4,
          'initial_RB_order': 0,
          'enrichment_target_error': 1.,
          'marking_doerfler_theta': 0.8,
          'marking_max_age': 2,
          'grid_type': 'alu'}

mpi_comm = MPI.COMM_WORLD
grid_and_problem_data = init_grid_and_problem(config, mpi_comm=mpi_comm)
grid = grid_and_problem_data['grid']
# grid.visualize('grid', False)

solver_options = {'max_iter': '400', 'precision': '1e-10', 'post_check_solves_system': '1e-5', 'type': 'bicgstab.ilut',
 'verbose': '0', 'preconditioner.iterations': '2', 'preconditioner.relaxation_factor': '1.0', }

LRBMS_d, data = discretize(grid_and_problem_data, solver_options={'inverse' :solver_options}, mpi_comm=mpi_comm)
block_space = data['block_space']
# LRBMS_d.disable_logging()
logger.debug('FULL OP DIM {}x{}'.format(LRBMS_d.operator.source.dim, LRBMS_d.operator.range.dim))

logger.info('estimating some errors:')
errors = []
for mu in np.linspace(grid_and_problem_data['parameter_range'][0],
                      grid_and_problem_data['parameter_range'][1],
                      3):

    pass
for mu in (grid_and_problem_data['parameter_range'][0],):
    mu = LRBMS_d.parse_parameter(mu)
    logger.info('  {}: '.format(mu))
    U = LRBMS_d.solve(mu, inverse_options=solver_options)
    LRBMS_d.visualize(U, filename='high_solution_{}.vtu'.format((mu['diffusion'])))
    estimate = LRBMS_d.estimate(U, mu=mu)
    # logger.info(estimate)
    errors.append(estimate)
# The estimator: either we
#  (i)  use the offline/online decomposable estimator (large offline computational effort, instant online estimation); or we
#  (ii) use the high-dimensional estimator (no offline effort, medium online effort).

red_solver_options = solver_options.copy()
red_solver_options['mpi_comm'] = mpi_comm
red_solver_options['type'] = 'mpi-manual_direct'
reductor = LRBMSReductor(
    LRBMS_d,
    products=[LRBMS_d.operators['local_energy_dg_product_{}'.format(ii)] for ii in range(block_space.num_blocks)],
    order=config['initial_RB_order']
)


logger.info('adding some global solution snapshots to reduced basis ...')
for mu in (grid_and_problem_data['mu_min'], grid_and_problem_data['mu_max']):
    U = LRBMS_d.solve(mu)
    try:
        reductor.extend_basis(U)
    except ExtensionError as e:
        logger.error(e)
logger.info('')
logger.debug('Bases count {} '.format(len(reductor.bases)))
# logger.debug(pprint.pformat(reductor.bases))
with np.printoptions(threshold=np.inf):
    logger.debug('DOM 0\n' + pprint.pformat(reductor.bases['domain_0'].data))
sys.exit(0)


with logger.block('reducing ...') as _:
    rd = reductor.reduce()

logger.debug('RED  OP DIM {}x{}'.format(rd.operator.source.dim, rd.operator.range.dim))

# raise RuntimeError('STOP')

with logger.block('estimating some reduced errors:') as _:
    # for mu in (grid_and_problem_data['mu_min'], grid_and_problem_data['mu_max']):
    for mu in (grid_and_problem_data['mu_min'], ):
        mu = rd.parse_parameter(mu)
        logger.info('{} ... '.format(mu))
        U = rd.solve(mu, inverse_options=red_solver_options)
        ur = reductor.reconstruct(U)
        LRBMS_d.visualize(ur, filename='recons_'+str(mu['diffusion']))
        estimate = rd.estimate(U, mu=mu)
        logger.info('reduced {}'.format(estimate))
        U = LRBMS_d.solve(mu)
        estimate = LRBMS_d.estimate(ur, mu=mu)
        logger.info('high  recon  {}'.format(estimate))
        estimate = LRBMS_d.estimate(U, mu=mu)
        logger.info('high  full   {}'.format(estimate))
        diff = U-ur
        LRBMS_d.visualize(diff, filename='diff_recon')

logger.info('')

# logger.info('online phase:')
# online_adaptive_LRBMS = AdaptiveEnrichment(grid_and_problem_data, LRBMS_d, block_space,
#                                            reductor, rd, config['enrichment_target_error'],
#                                            config['marking_doerfler_theta'],
#                                            config['marking_max_age'])
# for mu in rd.parameter_space.sample_randomly(20):
#     U, _, _ = online_adaptive_LRBMS.solve(mu)

# logger.info('')
# logger.info('local basis sizes:')
# for name, basis in online_adaptive_LRBMS.reductor.bases.items():
#     logger.info('{}: {}'.format(name, len(basis)))
# logger.info('finished')



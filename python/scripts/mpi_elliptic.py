#!/usr/bin/env python


import numpy as np
np.seterr(all='raise')

from pymor.core.exceptions import ExtensionError
from pymor.core.logger import getLogger, set_log_levels
set_log_levels({'online_adaptive_lrbms': 'DEBUG',
                'OS2015_academic_problem': 'INFO',
                'discretize_elliptic_block_swipdg': 'INFO',
                'offline': 'INFO',
                'online_enrichment': 'INFO',
                'lrbms': 'INFO'})
logger = getLogger('online_adaptive_lrbms.online_adaptive_lrbms')
from pymor.discretizations.basic import StationaryDiscretization

from dune.pylrbms.OS2015_academic_problem import init_grid_and_problem
# from local_thermalblock_problem import init_grid_and_problem
from dune.pylrbms.discretize_elliptic_swipdg import discretize

# max discretization error, to derive enrichment_target_error
# ===========================================================
# OS2015_academic_problem
# [4, 4], 2, [2, 2], 4: 0.815510144764
# [6, 6], 4, [6, 6], 4: 3.03372753518

# local_thermalblock_problem
# [6, 6], 4, [6, 6], 4: 0.585792065793
# ===========================================================

config = {'num_subdomains': None,
          'half_num_fine_elements_per_subdomain_and_dim': 2,
          'initial_RB_order': 0,
          'enrichment_target_error': 1.,
          'marking_doerfler_theta': 0.8,
          'marking_max_age': 2,
          'grid_type': 'alu'}


grid_and_problem_data = init_grid_and_problem(config)
grid = grid_and_problem_data['grid']
# grid.visualize('grid', False)

d, data = discretize(grid_and_problem_data, 1)
space = data['space']

sol = d.solve(0)

d.visualize(sol, filename='foo')

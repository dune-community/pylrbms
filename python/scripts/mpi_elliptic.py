#!/usr/bin/env python


import numpy as np
import mpi4py
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
from dune.xt.common import logging
logging.create(63)
from pymor.discretizations.basic import StationaryDiscretization

from dune.pylrbms.OS2015_academic_problem import init_grid_and_problem
# from dune.pylrbms.non_parametric_problem import init_grid_and_problem
# from local_thermalblock_problem import init_grid_and_problem
# from dune.pylrbms.discretize_elliptic_swipdg import discretize
from dune.pylrbms.discretize_elliptic_block_swipdg import discretize

# max discretization error, to derive enrichment_target_error
# ===========================================================
# OS2015_academic_problem
# [4, 4], 2, [2, 2], 4: 0.815510144764
# [6, 6], 4, [6, 6], 4: 3.03372753518

# local_thermalblock_problem
# [6, 6], 4, [6, 6], 4: 0.585792065793
# ===========================================================

config = {'num_subdomains': [2,2],
          'half_num_fine_elements_per_subdomain_and_dim': 4,
          'initial_RB_order': 0,
          'enrichment_target_error': 1.,
          'marking_doerfler_theta': 0.8,
          'marking_max_age': 2,
          'grid_type': 'alu'}


grid_and_problem_data = init_grid_and_problem(config)
grid = grid_and_problem_data['grid']
grid.visualize('grid', True)

mpi_comm = mpi4py.MPI.COMM_WORLD
solver_options = {'max_iter': '400', 'precision': '1e-10', 'post_check_solves_system': '1e-5', 'type': 'bicgstab.ilut',
 'verbose': '4', 'preconditioner.iterations': '2', 'preconditioner.relaxation_factor': '1.0', }
d, data = discretize(grid_and_problem_data, solver_options={'inverse' :solver_options}, mpi_comm=mpi_comm)

mu = {'diffusion': 0.5}
sol = d.solve(mu)

d.visualize(sol, filename='foo')


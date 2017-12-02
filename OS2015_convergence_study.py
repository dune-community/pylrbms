#!/usr/bin/env python

from functools import partial

import OS2015_academic_problem
from convergence_study import EllipticStudy
from discretize_elliptic_swipdg import discretize as discretize_elliptic_swipdg


OS2015_study = EllipticStudy(OS2015_academic_problem.init_grid_and_problem,
                             partial(discretize_elliptic_swipdg, polorder=1),
                             {'num_coarse_grid_elements': [4, 4],
                              'num_grid_refinements': 2,
                              'num_grid_subdomains': [2, 2],
                              'num_grid_oversampling_layers': 0},
                             mu=1)
OS2015_study.run(('h', 'elliptic_mu_bar'))


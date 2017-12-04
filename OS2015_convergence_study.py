#!/usr/bin/env python

from functools import partial

from OS2015_academic_problem import init_grid_and_problem
from convergence_study import StationaryEocStudy
# from discretize_elliptic_swipdg import discretize
from discretize_elliptic_block_swipdg import discretize


def refine(coarse_cfg):
    cfg = coarse_cfg.copy()
    cfg['num_grid_refinements'] += 2
    cfg['num_grid_subdomains'] = [s*2 for s in coarse_cfg['num_grid_subdomains']]
    return cfg


print('M. Ohlberger, F. Schindler, 2015, Error control for the Localized Reduced Basis')
print('                                  Multiscale method with adaptive on-line enrichment')
print('====================================================================================')
print()

OS2015_study = StationaryEocStudy(
        init_grid_and_problem,
        discretize,
        {'num_coarse_grid_elements': [4, 4],
         'num_grid_refinements': 2,
         'num_grid_subdomains': [2, 2],
         'num_grid_oversampling_layers': 1},
        refine,
        mu=1)
OS2015_study.indicators = ('eta_nc', 'eta_r', 'eta_df')

print('p. A2885, Table 1, all but \'eta_r\' and \'eff.\' columns')
OS2015_study.run(('h', 'elliptic_mu_bar', 'eta_nc', 'eta_df'))
print()

print('p. A2886, Table 2, \'eta_r\' column')
OS2015_study.run(('h', 'eta_r'))

print()


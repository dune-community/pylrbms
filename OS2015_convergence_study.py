#!/usr/bin/env python

from functools import partial

from OS2015_academic_problem import init_grid_and_problem
from EOC import StationaryEocStudy


def refine(coarse_cfg):
    cfg = coarse_cfg.copy()
    cfg['num_grid_refinements'] += 2
    cfg['num_grid_subdomains'] = [s*2 for s in coarse_cfg['num_grid_subdomains']]
    return cfg


def discretize(grid_and_problem_data):
    from discretize_elliptic_block_swipdg import discretize

    d, data = discretize(grid_and_problem_data)
    return d, {'block_space': data['block_space'], 'unblock': d.unblock}


print('M. Ohlberger, F. Schindler, 2015, Error control for the Localized Reduced Basis')
print('                                  Multiscale method with adaptive on-line enrichment')
print('====================================================================================')
print('Block SWIPDG P1')
print()

OS2015_all_mus_equal_study = StationaryEocStudy(
        init_grid_and_problem,
        discretize,
        {'num_coarse_grid_elements': [4, 4],
         'num_grid_refinements': 2,
         'num_grid_subdomains': [2, 2],
         'num_grid_oversampling_layers': 1},
        refine,
        mu=1)

print('p. A2885, Table 1, all but \'eta_r\' and \'eff.\' columns')
OS2015_all_mus_equal_study.run(('h', 'elliptic_mu_bar', 'eta_nc', 'eta_df'))
print()

print('p. A2886, Table 2, \'eta_r\' and \'eta\' (mu_hat=1)')
OS2015_all_mus_equal_study.run(('h', 'eta_r', 'eta'))
print()

print('p. A2886, Table 2, \'eta_df\' and \'eta\' (mu_hat=0.1)')
StationaryEocStudy(
        partial(init_grid_and_problem, mu_bar=1, mu_hat=0.1),
        discretize,
        {'num_coarse_grid_elements': [4, 4],
         'num_grid_refinements': 2,
         'num_grid_subdomains': [2, 2],
         'num_grid_oversampling_layers': 1},
        refine,
        mu=1).run(('h', 'eta_df', 'eta'))
print()

print('p. A2886, Table 3')
StationaryEocStudy(
        partial(init_grid_and_problem, mu_bar=0.1, mu_hat=0.1),
        discretize,
        {'num_coarse_grid_elements': [4, 4],
         'num_grid_refinements': 2,
         'num_grid_subdomains': [2, 2],
         'num_grid_oversampling_layers': 1},
        refine,
        mu=1).run(('h', 'elliptic_mu_bar', 'eta_nc', 'eta'))
print()


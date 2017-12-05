#!/usr/bin/env python

from functools import partial

from thermalblock_problem import init_grid_and_problem
from EOC import InstationaryEocStudy


def refine(coarse_cfg):
    cfg = coarse_cfg.copy()
    cfg['num_grid_refinements'] += 2
    cfg['num_grid_subdomains'] = [s*2 for s in coarse_cfg['num_grid_subdomains']]
    cfg['dt'] = 0.1 * init_grid_and_problem(cfg)['grid'].max_entity_diameter()
    return cfg


def discretize(grid_and_problem_data, T, nt):
    from discretize_parabolic_block_swipdg import discretize

    d, data = discretize(grid_and_problem_data, T, nt)
    return d, {'block_space': data['block_space'], 'unblock': d.unblock}


base_cfg = {'num_coarse_grid_elements': [4, 4],
            'num_grid_refinements': 2,
            'num_grid_subdomains': [2, 2],
            'num_grid_oversampling_layers': 1,
            'T': 1}
base_cfg['dt'] = 0.1 * init_grid_and_problem(base_cfg)['grid'].max_entity_diameter()

reference_cfg = base_cfg.copy()
for level in range(InstationaryEocStudy.max_levels + 1):
    reference_cfg = refine(reference_cfg)
reference_cfg['dt'] = 0.1 * init_grid_and_problem(reference_cfg)['grid'].max_entity_diameter()

mu = (1, 1, 1, 1)
print('Thermalblock, mu={}, Block SWIPDG P1, dt = 0.1*h'.format(mu))
print('===========================================================')
print()

parabolic_study = InstationaryEocStudy(
        init_grid_and_problem,
        discretize,
        base_cfg,
        refine,
        reference_cfg,
        mu=mu)
parabolic_study.run(('h', 'eta_nc', 'eta_r', 'eta_df', 'R_T', 'partial_t_nc'))
print()


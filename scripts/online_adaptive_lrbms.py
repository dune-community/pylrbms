#!/usr/bin/env python

import numpy as np

from dune.gdt import project

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.bindings.dunext import DuneXTVector
from pymor.core.exceptions import ExtensionError
from pymor.reductors.system import GenericRBSystemReductor
from pymor.vectorarrays.list import ListVectorArray

from OS2015_academic_problem import (
        make_expression_function_1x1,
        init_grid_and_problem,
)
from discretize_elliptic import (
        Vector,
        discretize,
        make_discrete_function,
)


config = {'num_coarse_grid_elements': [4, 4],
          'num_grid_refinements': 2,
          'num_grid_subdomains': [2, 2],
          'num_grid_oversampling_layers': 4} # num_grid_oversampling_layers has to exactly cover one subdomain!

grid_and_problem_data = init_grid_and_problem(config)
grid = grid_and_problem_data['grid']

d, block_space = discretize(grid_and_problem_data)

print('estimating some errors:')
errors = []
for mu in (grid_and_problem_data['mu_min'], grid_and_problem_data['mu_max']):
    mu = d.parse_parameter(mu)
    print('  {}: '.format(mu), end='', flush=True)
    U = d.solve(mu)
    estimate = d.estimate(U, mu=mu)
    print(estimate)
    errors.append(estimate)

print('initializing local reduced bases with DG shape functions ... ', end='')

U = d.solution_space.empty()
reductor = GenericRBSystemReductor(d, products=[d.operators['local_energy_dg_product_{}'.format(ii)]
                                                for ii in range(block_space.num_blocks)])
for ii in range(block_space.num_blocks):
    local_space = block_space.local_space(ii)
    # order 0 basis
    reductor.extend_basis_local(ListVectorArray([DuneXTVector(Vector(local_space.size(), 1.)), ],
                                                U._blocks[ii].space))
    # order 1 basis
    tmp_discrete_function = make_discrete_function(local_space)
    for expression in ('x[0]', 'x[1]', 'x[0]*x[1]'):
        func = make_expression_function_1x1(grid, 'x', expression, order=2)
        project(func, tmp_discrete_function)
        reductor.extend_basis_local(ListVectorArray([DuneXTVector(tmp_discrete_function.vector_copy()), ],
                                                    U._blocks[ii].space))
    del tmp_discrete_function

print('done')

print('adding some global solution snapshots to reduced basis:')
for mu in (grid_and_problem_data['mu_min'], grid_and_problem_data['mu_max']):
    U = d.solve(mu)
    try:
        reductor.extend_basis(U)
    except ExtensionError:
        pass

print('reducing:')

rd = reductor.reduce()
rd = rd.with_(estimator=d.estimator)

print('estimating some reduced errors:')
mus = [grid_and_problem_data['mu_min'], grid_and_problem_data['mu_max']]
mus.extend(d.parameter_space.sample_randomly(3))
mus = [mu if isinstance(mu, tuple) else mu['diffusion'] for mu in mus]
mus.sort()
reduced_errors = []
for mu in mus:
    mu = d.parse_parameter(mu)
    print('  {}: '.format(mu), end='', flush=True)
    U = rd.solve(mu)
    estimate = rd.estimate(U, mu=mu)
    print(estimate)
    reduced_errors.append(estimate)


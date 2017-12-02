#!/usr/bin/env python

import numpy as np

from dune.gdt import (
        make_discrete_function,
        prolong,
        RS2017_residual_indicator_subdomain_diameter as subdomain_diameter,
        )

from pymor.core.logger import set_log_levels
set_log_levels({'pymor.discretizations.basic': 'WARN',})

from discretize_elliptic_swipdg import discretize as discretize_elliptic_swipdg


class EocStudy(object):

    level_info_title = None
    accuracies = None
    EOC_accuracy = None
    norms = None
    max_levels = None
    data = {}

    def compute(self, level):
        pass

    def level_info(self, level):
        pass

    def accuracy(self, level, id):
        pass

    def norm(self, level, id):
        pass

    def run(self, only_these = None):

        def lfill(id_, len_):
            if len(id_) == len_:
                return id_
            if len(id_) > len_:
                return id_[:(len_ - 1)] + '.'
            elif len(id_) < len_:
                return ' '*(len_ - len(id_)) + id_

        def cfill(id_, len_):
            assert len(id_) < len_
            rpadd = int((len_ - len(id_))/2)
            return ' '*(len_ - rpadd - len(id_)) + id_ + ' '*rpadd

        column_width = 8

        # build header
        #   discretization
        h1 = ' '
        d1 = '-'*(len(self.level_info_title) + 2)
        h2 = ' ' + self.level_info_title + ' '
        delim = '-'*(len(self.level_info_title) + 2)
        for id in self.accuracies:
            if not only_these or id in only_these:
                h2 += '| ' + lfill(id, column_width) + ' '
                d1 += '-'*11
                delim += '+' + '-'*10
        h1 += cfill('discretization', len(h2) - 1)
        #   norms
        for id in self.norms:
            if not only_these or id in only_these:
                h1 += '|' + cfill(id, 21)
                d1 += '+' + '-'*21
                h2 += '| ' + lfill('norm', column_width) + ' | ' + lfill('EOC', column_width) + ' '
                delim += '+' + '-'*10 + '+' + '-'*10
        # print header
        print('=' * len(h1))
        print(h1)
        print(d1)
        print(h2)
        print(delim.replace('-', '=', -1))
        # print levels
        for level in range(self.max_levels + 1):
            if not level in self.data:
                self.data[level] = {}
            self.compute(level)
            print(' ' + cfill(self.level_info(level), len(self.level_info_title)) + ' ', end='')
            if not 'accuracy' in self.data[level]:
                self.data[level]['accuracy'] = {}
            for id in self.accuracies:
                self.data[level]['accuracy'][id] = self.accuracy(level, id)
                if not only_these or id in only_these:
                    print('| ' + lfill('{:.2e}'.format(self.data[level]['accuracy'][id]), 8) + ' ', end='')
            if not 'norm' in self.data[level]:
                self.data[level]['norm'] = {}
            for id in self.norms:
                if not only_these or id in only_these:
                    self.data[level]['norm'][id] = self.norm(level, id)
                    print('| ' + lfill('{:.2e}'.format(self.data[level]['norm'][id]), 8) + ' ', end='')
                    if level == 0:
                        print('| ' + lfill('----', column_width) + ' ', end='')
                    else:
                        EOC = (  np.log(self.data[level]['norm'][id]
                                / self.data[level - 1]['norm'][id])
                              /np.log(self.data[level]['accuracy'][self.EOC_accuracy]
                                / self.data[level - 1]['accuracy'][self.EOC_accuracy]))
                        print('| ' + lfill('{:.2f}'.format(EOC), column_width) + ' ', end='')
            print()
            if level < self.max_levels:
                print(delim)


class EllipticStudy(EocStudy):

    level_info_title = '|grid|/|Grid|'
    accuracies = ('h', 'H')
    EOC_accuracy = 'h'
    norms = ('L2', 'elliptic_mu_bar')
    indicators = ('eta_nc',)
    max_levels = 3
    _grid_and_problem_data, _d, _d_data, _solution = {}, {}, {}, {}
    _config = {}

    def __init__(self, gp_initializer, disc, base_cfg, mu, p_ref=2):
        self._grid_and_problem_initializer = gp_initializer
        self._discretizer = disc
        self._base_config = base_cfg
        self.mu = mu
        for level in range(self.max_levels + 1):
            self._config[level] = self._base_config.copy()
            self._config[level]['num_grid_refinements'] += 2*level
            self._config[level]['num_grid_subdomains']  = [s*2*level for s in self._base_config['num_grid_subdomains']]
            self._config[level]['num_grid_oversampling_layers'] *= 2*level
        # compute reference solution
        self._config[-1] = self._config[self.max_levels].copy()
        self._grid_and_problem_data[-1] = self._grid_and_problem_initializer(self._config[-1])
        self._d[-1], self._d_data[-1] = discretize_elliptic_swipdg(self._grid_and_problem_data[-1], p_ref)
        space = self._d_data[-1]['space']
        self._solution[-1] = self._d[-1].solve(self.mu)._list[0].impl
        self._solution[-1] = make_discrete_function(space, self._solution[-1])
        # prepare error products
        self._l2_product = self._d[-1].operators['l2'].matrix
        self._elliptic_mu_bar_product = self._d[-1].operators['elliptic_mu_bar'].matrix

    def compute(self, level):
        assert level <= self.max_levels
        self._grid_and_problem_data[level] = self._grid_and_problem_initializer(self._config[level])
        self._d[level], self._d_data[level] = self._discretizer(self._grid_and_problem_data[level])
        if 'block_space' in self._d_data[level]:
            coarse_solution = self._d[level].unblock(self._d[level].solve(self.mu))._list[0].impl
            coarse_solution = make_discrete_function(self._d_data[level]['block_space'], coarse_solution)
        else:
            coarse_solution = self._d[level].solve(self.mu)._list[0].impl
            coarse_solution = make_discrete_function(self._d_data[level]['space'], coarse_solution)
        self._solution[level] = make_discrete_function(self._d_data[-1]['space'])
        prolong(coarse_solution, self._solution[level])

    def level_info(self, level):
        assert level <= self.max_levels
        grid = self._grid_and_problem_data[level]['grid']
        return str(grid.num_elements) + '/' + str(grid.num_subdomains)

    def accuracy(self, level, id):
        assert level <= self.max_levels
        grid = self._grid_and_problem_data[level]['grid']
        if id == 'h':
            return grid.max_entity_diameter()
        elif id == 'H':
            diam = -1
            for ss in range(grid.num_subdomains):
                diam = max(diam, subdomain_diameter(grid, ss))
            return diam
        else:
            assert False

    def norm(self, level, id):
        diff = self._solution[-1].vector_copy() - self._solution[level].vector_copy()
        if id == 'L2':
            return np.sqrt(diff * (self._l2_product * diff))
        elif id == 'elliptic_mu_bar':
            return np.sqrt(diff * (self._elliptic_mu_bar_product * diff))
        else:
            assert False


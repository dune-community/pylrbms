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


class EocStudy:

    level_info_title = None
    accuracies = None
    norms = None
    indicators = None
    estimates = None
    max_levels = None
    data = None

    def solve(self, level):
        pass

    def level_info(self, level):
        pass

    def accuracy(self, level, id):
        pass

    def compute_norm(self, level, id):
        pass

    def compute_indicator(self, level, id):
        pass

    def compute_estimate(self, level, id):
        pass

    def run(self, only_these = None):

        actual_accuracies = self.accuracies if not only_these else tuple(id for id in self.accuracies if id in only_these)
        actual_norms = self.norms if not only_these else tuple(id for id in self.norms if id in only_these)
        actual_indicators = self.indicators if not only_these else tuple(id for id in self.indicators if id in only_these)
        actual_estimates = self.estimates if not only_these else tuple(ids for ids in self.estimates if ids[0] in only_these)

        column_width = 8
        if len(actual_accuracies) > 1:
            eoc_column_width = 8
        else:
            eoc_column_width = 4
        print_full_estimate = False

        def lfill(id_, len_):
            if len(id_) == len_:
                return id_
            if len(id_) > len_:
                return id_[:(len_ - 1)] + '.'
            elif len(id_) < len_:
                return ' '*(len_ - len(id_)) + id_

        def cfill(id_, len_):
            if len(id_) > len_ - 1:
                return id_[:(len_ - 1)] + '.'
            rpadd = int((len_ - len(id_))/2)
            return ' '*(len_ - rpadd - len(id_)) + id_ + ' '*rpadd

        def compute_eoc(quantity_old, quantity_new, level, accuracy_id):
            if np.allclose(quantity_old, 0):
                return lfill('inf', eoc_column_width)
            else:
                accuracy_old = self.data[level - 1]['accuracy'][accuracy_id]
                accuracy_new = self.data[level]['accuracy'][accuracy_id]
                return lfill('{:.2f}'.format(np.log(quantity_new / quantity_old) / np.log(accuracy_new / accuracy_old)),
                             eoc_column_width)


        # build header
        #   discretization
        h1 = ' '
        d1 = '-'*(len(self.level_info_title) + 2)
        h2 = ' ' + self.level_info_title + ' '
        delim = '-'*(len(self.level_info_title) + 2)
        for id in self.accuracies:
            if not only_these or id in only_these:
                h2 += '| ' + lfill(id, column_width) + ' '
                d1 += '-'*(column_width + 3)
                delim += '+' + '-'*(column_width + 2)
        h1 += cfill('discretization', len(self.level_info_title) + len(actual_accuracies)*(column_width + 3)) + ' '
        #   norms
        for id in actual_norms:
            h1 += '| ' + cfill(id, column_width + len(actual_accuracies)*(eoc_column_width + 3)) + ' '
            d1 += '+' + '-'*(column_width + 2 + len(actual_accuracies)*(eoc_column_width + 3))
            h2 += '| ' + lfill('norm', column_width) + ' '
            delim += '+' + '-'*(column_width + 2)
            for acc_id in actual_accuracies:
                h2 += '| ' + lfill('EOC' if len(actual_accuracies) == 1 else 'EOC ({})'.format(acc_id), eoc_column_width) + ' '
                delim += '+' + '-'*(eoc_column_width + 2)
        #   indicators
        for id in actual_indicators:
            h1 += '| ' + cfill(id, column_width + len(actual_accuracies)*(eoc_column_width + 3)) + ' '
            d1 += '+' + '-'*(column_width + 2 + len(actual_accuracies)*(eoc_column_width + 3))
            h2 += '| ' + lfill('indicator', column_width) + ' '
            delim += '+' + '-'*(column_width + 2)
            for acc_id in actual_accuracies:
                h2 += '| ' + lfill('EOC' if len(actual_accuracies) == 1 else 'EOC ({})'.format(acc_id), eoc_column_width) + ' '
                delim += '+' + '-'*(eoc_column_width + 2)
        #   estimates
        for id, _ in actual_estimates:
            if print_full_estimate:
                h1 += '| ' + cfill(id, column_width + (len(actual_accuracies) + 1)*(eoc_column_width + 3)) + ' '
            else:
                h1 += '| ' + cfill(id, (len(actual_accuracies) + 1)*(eoc_column_width + 3) - 2) + ' '
            if print_full_estimate:
                d1 += '+' + '-'*(column_width + 2)
                h2 += '| ' + lfill('estimate', column_width) + ' '
                delim += '+' + '-'*(column_width + 2)
            d1 += '+' + '-'*(eoc_column_width + 2)
            h2 += '| ' + lfill('eff.', eoc_column_width) + ' '
            delim += '+' + '-'*(eoc_column_width + 2)
            for acc_id in actual_accuracies:
                d1 += '+' + '-'*(eoc_column_width + 2)
                h2 += '| ' + lfill('EOC' if len(actual_accuracies) == 1 else 'EOC ({})'.format(acc_id), eoc_column_width) + ' '
                delim += '+' + '-'*(eoc_column_width + 2)
        # print header
        print('=' * len(h1))
        print(h1)
        print(d1)
        print(h2)
        print(delim.replace('-', '=', -1))
        # print levels
        for level in range(self.max_levels + 1):
            # level info
            if not level in self.data:
                self.data[level] = {}
            self.solve(level)
            print(' ' + cfill(self.level_info(level), len(self.level_info_title)) + ' ', end='')
            # accuracies
            if not 'accuracy' in self.data[level]:
                self.data[level]['accuracy'] = {}
            for id in self.accuracies:
                self.data[level]['accuracy'][id] = self.accuracy(level, id)
                if not only_these or id in only_these:
                    print('| ' + lfill('{:.2e}'.format(self.data[level]['accuracy'][id]), column_width) + ' ', end='')
            # norms
            if not 'norm' in self.data[level]:
                self.data[level]['norm'] = {}
            for id in actual_norms:
                self.data[level]['norm'][id] = self.compute_norm(level, id)
                print('| ' + lfill('{:.2e}'.format(self.data[level]['norm'][id]), column_width) + ' ', end='')
                for acc_id in actual_accuracies:
                    if level == 0:
                        print('| ' + lfill('----', eoc_column_width) + ' ', end='')
                    else:
                        print('| ' + compute_eoc(self.data[level - 1]['norm'][id],
                                                 self.data[level]['norm'][id],
                                                 level, acc_id) + ' ', end='')
            # indicators
            if not 'indicator' in self.data[level]:
                self.data[level]['indicator'] = {}
            for id in actual_indicators:
                self.data[level]['indicator'][id] = self.compute_indicator(level, id)
                print('| ' + lfill('{:.2e}'.format(self.data[level]['indicator'][id]), column_width) + ' ', end='')
                for acc_id in actual_accuracies:
                    if level == 0:
                        print('| ' + lfill('----', eoc_column_width) + ' ', end='')
                    else:
                        print('| ' + compute_eoc(self.data[level - 1]['indicator'][id],
                                                 self.data[level]['indicator'][id],
                                                 level, acc_id) + ' ', end='')
            if not 'estimate' in self.data[level]:
                self.data[level]['estimate'] = {}
            for estimate_id, norm_id in actual_estimates:
                self.data[level]['estimate'][estimate_id] = self.compute_estimate(level, estimate_id)
                if not norm_id in self.data[level]['norm']:
                    self.data[level]['norm'][norm_id] = self.compute_norm(level, norm_id)
                if print_full_estimate:
                    print('| ' + lfill('{:.2e}'.format(self.data[level]['estimate'][estimate_id]), column_width) + ' ', end='')
                print('| ' +
                        lfill('{:.2f}'.format(self.data[level]['estimate'][estimate_id]/self.data[level]['norm'][norm_id]),
                            eoc_column_width) + ' ', end='')
                for acc_id in actual_accuracies:
                    if level == 0:
                        print('| ' + lfill('----', eoc_column_width) + ' ', end='')
                    else:
                        print('| ' + compute_eoc(self.data[level - 1]['estimate'][estimate_id],
                                                 self.data[level]['estimate'][estimate_id],
                                                 level, acc_id) + ' ', end='')
            print()
            if level < self.max_levels:
                print(delim)


class StationaryEocStudy(EocStudy):

    level_info_title = '|grid|/|Grid|'
    accuracies = ('h', 'H')
    EOC_accuracy = 'h'
    norms = ('L2', 'elliptic_mu_bar')
    indicators = None
    estimates = (('eta', 'elliptic_mu_bar'), )
    max_levels = 3

    def __init__(self, gp_initializer, disc, base_cfg, refine, mu, p_ref=2):
        self.data = {}
        (self._grid_and_problem_data, self._d, self._d_data, self._solution, self._solution_as_reference, self._config,
         self._cache) = {}, {}, {}, {}, {}, {}, {}
        self._grid_and_problem_initializer = gp_initializer
        self._discretizer = disc
        self.mu = mu
        self._config[0] = base_cfg.copy()
        for level in range(1, self.max_levels + 1):
            self._config[level] = refine(self._config[level - 1])
        # compute reference solution
        self._config[-1] = self._config[self.max_levels].copy()
        self._grid_and_problem_data[-1] = self._grid_and_problem_initializer(self._config[-1])
        self._d[-1], self._d_data[-1] = discretize_elliptic_swipdg(self._grid_and_problem_data[-1], p_ref)
        space = self._d_data[-1]['space']
        mu = self._d[-1].parse_parameter(self.mu)
        self._solution[-1] = self._d[-1].solve(mu)._list[0].impl
        self._solution[-1] = make_discrete_function(space, self._solution[-1])
        # prepare error products
        self._l2_product = self._d[-1].operators['l2'].matrix
        self._elliptic_mu_bar_product = self._d[-1].operators['elliptic_mu_bar'].matrix

    def solve(self, level):
        assert level <= self.max_levels
        self._grid_and_problem_data[level] = self._grid_and_problem_initializer(self._config[level])
        self._d[level], self._d_data[level] = self._discretizer(self._grid_and_problem_data[level])
        mu = self._d[level].parse_parameter(self.mu)
        self._solution[level] = self._d[level].solve(mu)
        if 'block_space' in self._d_data[level]:
            coarse_solution = make_discrete_function(self._d_data[level]['block_space'],
                                                     self._d[level].unblock(self._solution[level])._list[0].impl)
        else:
            coarse_solution = make_discrete_function(self._d_data[level]['space'],
                                                     self._solution[level]._list[0].impl)
        self._solution_as_reference[level] = make_discrete_function(self._d_data[-1]['space'])
        prolong(coarse_solution, self._solution_as_reference[level])

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

    def compute_norm(self, level, id):
        diff = self._solution[-1].vector_copy() - self._solution_as_reference[level].vector_copy()
        if id == 'L2':
            return np.sqrt(diff * (self._l2_product * diff))
        elif id == 'elliptic_mu_bar':
            return np.sqrt(diff * (self._elliptic_mu_bar_product * diff))
        else:
            assert False

    def _compute_estimates(self, level):
        if not level in self._cache:
            mu = self._d[level].parse_parameter(self.mu)
            eta, (eta_ncs, eta_rs, eta_dfs), _ = self._d[level].estimate(self._solution[level], mu=mu, decompose=True)
            self._cache[level] = {
                    'eta_nc': np.linalg.norm(eta_ncs),
                    'eta_df': np.linalg.norm(eta_dfs),
                    'eta_r': np.linalg.norm(eta_rs),
                    'eta': eta[0]}

    def compute_indicator(self, level, id):
        self._compute_estimates(level)
        return self._cache[level][id]

    def compute_estimate(self, level, id):
        self._compute_estimates(level)
        return self._cache[level][id]


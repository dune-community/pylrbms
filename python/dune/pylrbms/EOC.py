#!/usr/bin/env python

import sys
import numpy as np
np.warnings.filterwarnings('ignore')

import itertools

from dune.gdt.discretefunction import make_discrete_function
from dune.gdt import (
        prolong,
        )
from dune.gdt.playground.operators.rs2017 import RS2017_residual_indicator_subdomain_diameter as subdomain_diameter

from pymor.core.logger import set_log_levels
set_log_levels({'pymor.discretizations.basic': 'WARN',})
from pymor.grids.oned import OnedGrid

from dune.pylrbms.discretize_elliptic_swipdg import discretize as discretize_elliptic_swipdg
from dune.pylrbms.discretize_parabolic_swipdg import discretize as discretize_parabolic_swipdg


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

        self.accuracies = self.accuracies or []
        self.norms = self.norms or []
        self.indicators = self.indicators or []
        self.estimates = self.estimates or []
        actual_accuracies = self.accuracies if not only_these else tuple(id for id in self.accuracies if id in only_these)
        actual_norms = self.norms if not only_these and self.norms else tuple(id for id in self.norms if id in only_these)
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
            if len(id_) > len_:
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
            h2 += '| ' + lfill('value', column_width) + ' '
            delim += '+' + '-'*(column_width + 2)
            for acc_id in actual_accuracies:
                h2 += '| ' + lfill('EOC' if len(actual_accuracies) == 1 else 'EOC ({})'.format(acc_id), eoc_column_width) + ' '
                delim += '+' + '-'*(eoc_column_width + 2)
        #   indicators
        for id in actual_indicators:
            h1 += '| ' + cfill(id, column_width + len(actual_accuracies)*(eoc_column_width + 3)) + ' '
            d1 += '+' + '-'*(column_width + 2 + len(actual_accuracies)*(eoc_column_width + 3))
            h2 += '| ' + lfill('value', column_width) + ' '
            delim += '+' + '-'*(column_width + 2)
            for acc_id in actual_accuracies:
                h2 += '| ' + lfill('EOC' if len(actual_accuracies) == 1 else 'EOC ({})'.format(acc_id), eoc_column_width) + ' '
                delim += '+' + '-'*(eoc_column_width + 2)
        #   estimates
        for id, _ in actual_estimates:
            if print_full_estimate:
                h1 += '| ' + cfill(id, column_width + (len(actual_accuracies) + 1)*(eoc_column_width + 3)) + ' '
            else:
                h1 += '| ' + cfill(id, (len(actual_accuracies) + 1)*(eoc_column_width + 3) - 3) + ' '
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
            sys.stdout.flush()
            # accuracies
            if not 'accuracy' in self.data[level]:
                self.data[level]['accuracy'] = {}
            for id in self.accuracies:
                self.data[level]['accuracy'][id] = self.accuracy(level, id)
                if not only_these or id in only_these:
                    print('| ' + lfill('{:.2e}'.format(self.data[level]['accuracy'][id]), column_width) + ' ', end='')
                    sys.stdout.flush()
            # norms
            if not 'norm' in self.data[level]:
                self.data[level]['norm'] = {}
            for id in actual_norms:
                self.data[level]['norm'][id] = self.compute_norm(level, id)
                print('| ' + lfill('{:.2e}'.format(self.data[level]['norm'][id]), column_width) + ' ', end='')
                sys.stdout.flush()
                for acc_id in actual_accuracies:
                    if level == 0:
                        print('| ' + lfill('----', eoc_column_width) + ' ', end='')
                        sys.stdout.flush()
                    else:
                        print('| ' + compute_eoc(self.data[level - 1]['norm'][id],
                                                 self.data[level]['norm'][id],
                                                 level, acc_id) + ' ', end='')
                        sys.stdout.flush()
            # indicators
            if not 'indicator' in self.data[level]:
                self.data[level]['indicator'] = {}
            for id in actual_indicators:
                self.data[level]['indicator'][id] = self.compute_indicator(level, id)
                print('| ' + lfill('{:.2e}'.format(self.data[level]['indicator'][id]), column_width) + ' ', end='')
                sys.stdout.flush()
                for acc_id in actual_accuracies:
                    if level == 0:
                        print('| ' + lfill('----', eoc_column_width) + ' ', end='')
                        sys.stdout.flush()
                    else:
                        print('| ' + compute_eoc(self.data[level - 1]['indicator'][id],
                                                 self.data[level]['indicator'][id],
                                                 level, acc_id) + ' ', end='')
                        sys.stdout.flush()
            if not 'estimate' in self.data[level]:
                self.data[level]['estimate'] = {}
            for estimate_id, norm_id in actual_estimates:
                self.data[level]['estimate'][estimate_id] = self.compute_estimate(level, estimate_id)
                if not norm_id in self.data[level]['norm']:
                    self.data[level]['norm'][norm_id] = self.compute_norm(level, norm_id)
                if print_full_estimate:
                    print('| ' + lfill('{:.2e}'.format(self.data[level]['estimate'][estimate_id]), column_width) + ' ', end='')
                    sys.stdout.flush()
                print('| ' +
                        lfill('{:.2f}'.format(self.data[level]['norm'][norm_id]/self.data[level]['estimate'][estimate_id]),
                            eoc_column_width) + ' ', end='')
                sys.stdout.flush()
                for acc_id in actual_accuracies:
                    if level == 0:
                        print('| ' + lfill('----', eoc_column_width) + ' ', end='')
                        sys.stdout.flush()
                    else:
                        print('| ' + compute_eoc(self.data[level - 1]['estimate'][estimate_id],
                                                 self.data[level]['estimate'][estimate_id],
                                                 level, acc_id) + ' ', end='')
                        sys.stdout.flush()
            print()
            if level < self.max_levels:
                print(delim)


class StationaryEocStudy(EocStudy):

    level_info_title = '|grid|/|Grid|'
    accuracies = ('h', 'H')
    norms = ('L2', 'elliptic_mu_bar')
    indicators = ('eta_nc', 'eta_r', 'eta_df')
    estimates = (('eta', 'elliptic_mu_bar'), )
    max_levels = 2

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
        self._config[-1] = self._config[self.max_levels].copy()
        self.p_ref = p_ref

    def solve(self, level):
        assert level <= self.max_levels
        self._grid_and_problem_data[level] = self._grid_and_problem_initializer(self._config[level])
        self._d[level], self._d_data[level] = self._discretizer(self._grid_and_problem_data[level])
        mu = self._d[level].parse_parameter(self.mu)
        self._solution[level] = self._d[level].solve(mu)

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
        self._compute_reference_solution()
        self._prolong_onto_reference(level)
        diff = self._solution[-1].vector_copy() - self._solution_as_reference[level].vector_copy()
        if id == 'L2':
            return np.sqrt(diff * (self._l2_product * diff))
        elif id == 'elliptic_mu_bar':
            return np.sqrt(diff * (self._elliptic_mu_bar_product * diff))
        else:
            assert False

    def compute_indicator(self, level, id):
        self._compute_estimates(level)
        return self._cache[level][id]

    def compute_estimate(self, level, id):
        self._compute_estimates(level)
        return self._cache[level][id]

    def _compute_reference_solution(self):
        if -1 in self._solution:
            return
        self._grid_and_problem_data[-1] = self._grid_and_problem_initializer(self._config[-1])
        self._d[-1], self._d_data[-1] = discretize_elliptic_swipdg(self._grid_and_problem_data[-1], self.p_ref)
        space = self._d_data[-1]['space']
        self._solution[-1] = self._d[-1].solve(self._d[-1].parse_parameter(self.mu))._list[0].impl
        self._solution[-1] = make_discrete_function(space, self._solution[-1])
        # prepare error products
        self._l2_product = self._d[-1].operators['l2'].matrix
        self._elliptic_mu_bar_product = self._d[-1].operators['elliptic_mu_bar'].matrix

    def _prolong_onto_reference(self, level):
        if level in self._solution_as_reference:
            return
        if 'reductor' in self._d_data[level]:
            reconstructed_soluion = self._d_data[level]['reductor'].reconstruct(self._solution[level])
        else:
            reconstructed_soluion = self._solution[level]
        if 'block_space' in self._d_data[level]:
            coarse_solution = make_discrete_function(self._d_data[level]['block_space'],
                                                     self._d_data[level]['unblock'](reconstructed_soluion)._list[0].impl)
        else:
            coarse_solution = make_discrete_function(self._d_data[level]['space'],
                                                     reconstructed_soluion._list[0].impl)
        self._solution_as_reference[level] = make_discrete_function(self._d_data[-1]['space'])
        prolong(coarse_solution, self._solution_as_reference[level])

    def _compute_estimates(self, level):
        if not level in self._cache:
            mu = self._d[level].parse_parameter(self.mu)
            eta, (eta_ncs, eta_rs, eta_dfs), _ = self._d[level].estimate(self._solution[level], mu=mu, decompose=True)
            self._cache[level] = {
                    'eta_nc': np.linalg.norm(eta_ncs),
                    'eta_df': np.linalg.norm(eta_dfs),
                    'eta_r': np.linalg.norm(eta_rs),
                    'eta': eta[0]}


class InstationaryEocStudy(EocStudy):

    level_info_title = '|grid|/|Grid|/nt'
    accuracies = ('h', 'H', 'dt')
    norms = tuple('{} - {}'.format(tpl[0], tpl[1])
                  for tpl in itertools.product(['L_oo', 'L2'], ['L2', 'elliptic_mu_bar']))
    indicators = ('eta_nc', 'eta_r', 'eta_df', 'R_T', 'partial_t_nc')
    estimates = (('eta', 'L2 - elliptic_mu_bar'), )
    max_levels = 2

    def __init__(self, gp_initializer, disc, base_cfg, refine, reference_cfg, mu, p_ref=2):
        self.data = {}
        (self._grid_and_problem_data, self._d, self._d_data, self._solution, self._solution_as_reference, self._config,
         self._cache) = {}, {}, {}, {}, {}, {}, {}
        self._grid_and_problem_initializer = gp_initializer
        self._discretizer = disc
        self.mu = mu
        self._config[0] = base_cfg.copy()
        for level in range(1, self.max_levels + 1):
            self._config[level] = refine(self._config[level - 1])
        self._config[-1] = reference_cfg
        self._T = self._config[0]['T']
        self.p_ref = p_ref

    def solve(self, level):
        assert level <= self.max_levels
        self._grid_and_problem_data[level] = self._grid_and_problem_initializer(self._config[level])
        T = self._T
        dt = self._config[level]['dt']
        self._d[level], self._d_data[level] = self._discretizer(
                self._grid_and_problem_data[level],
                T,
                int(T / dt) + 1)
        self._solution[level] = self._d[level].solve(self._d[level].parse_parameter(self.mu))

    def level_info(self, level):
        # return str(level)
        assert level <= self.max_levels
        grid = self._grid_and_problem_data[level]['grid']
        nt = len(self._solution[level]) - 1
        return str(grid.num_elements) + '/' + str(grid.num_subdomains) + '/' + str(nt)

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
        elif id == 'dt':
            return self._config[level]['dt']
        else:
            assert False

    def compute_norm(self, level, id):
        self._compute_reference_solution()
        self._prolong_onto_reference(level)
        diff = [ref.vector_copy() - sol.vector_copy()
                for ref, sol in zip(self._solution[-1], self._solution_as_reference[level])]
        time_norm_id, space_norm_id = id.split('-')
        time_norm_id, space_norm_id = time_norm_id.strip(), space_norm_id.strip()
        if space_norm_id == 'L2':
            def space_norm2(U):
                return U * (self._l2_product * U)
        elif space_norm_id == 'elliptic_mu_bar':
            def space_norm2(U):
                return U * (self._elliptic_mu_bar_product * U)
        else:
            assert False
        if time_norm_id == 'L_oo':
            return np.max(np.sqrt([space_norm2(U) for U in diff]))
        elif time_norm_id == 'L2':
            T = self._T
            nt = len(diff)
            time_integration_order = 1
            time_grid = OnedGrid(domain=(0., T), num_intervals=nt - 1)
            qq = time_grid.quadrature_points(0, order=time_integration_order)
            ww = time_grid.reference_element.quadrature(time_integration_order)[1] # weights
            integral = 0.
            for entity in np.arange(time_grid.size(0)):
                # get quadrature
                qq_e = qq[entity] # points
                ie = time_grid.integration_elements(0)[entity]
                # create shape function evaluations
                a = time_grid.centers(1)[entity]
                b = time_grid.centers(1)[entity + 1]
                SF = np.array((1./(a - b)*qq_e[..., 0] - b/(a - b), # P1 in time
                               1./(b - a)*qq_e[..., 0] - a/(b - a)))
                U_a = diff[entity]
                U_b = diff[entity + 1]
                values = np.zeros(len(qq_e))
                for ii in np.arange(len(qq_e)):
                    U_t = U_a.copy()
                    U_t.scal(SF[0][ii])
                    U_t.axpy(SF[1][ii], U_b)
                    values[ii] = space_norm2(U_t)
                integral += np.dot(values, ww)*ie
            return np.sqrt(integral)
        else:
            assert False

    def compute_indicator(self, level, id):
        self._compute_estimates(level)
        return self._cache[level][id]

    def compute_estimate(self, level, id):
        self._compute_estimates(level)
        return self._cache[level][id]

    def _compute_reference_solution(self):
        if -1 in self._solution:
            return
        self._grid_and_problem_data[-1] = self._grid_and_problem_initializer(self._config[-1])
        dt = self._config[-1]['dt']
        self._d[-1], self._d_data[-1] = discretize_parabolic_swipdg(
                self._grid_and_problem_data[-1],
                self._T,
                int(self._T / dt) + 1,
                self.p_ref)
        space = self._d_data[-1]['space']
        self._solution[-1] = self._d[-1].solve(self._d[-1].parse_parameter(self.mu))
        self._solution[-1] = tuple(make_discrete_function(space, vec.impl) for vec in self._solution[-1]._list)
        # prepare error products
        self._l2_product = self._d[-1].operators['l2'].matrix
        self._elliptic_mu_bar_product = self._d[-1].operators['elliptic_mu_bar'].matrix

    def _prolong_onto_reference(self, level):
        if level in self._solution_as_reference:
            return
        # reconstruct, if reduced
        if 'reductor' in self._d_data[level]:
            assert False # not yet implemented
            reconstructed_soluion = self._d_data[level]['reductor'].reconstruct(self._solution[level])
        else:
            reconstructed_soluion = self._solution[level]
        if 'block_space' in self._d_data[level]:
            coarse_solution = tuple(make_discrete_function(self._d_data[level]['block_space'], vec.impl)
                                    for vec in self._d_data[level]['unblock'](reconstructed_soluion)._list)
        else:
            coarse_solution = tuple(make_discrete_function(self._d_data[level]['space'], vec.impl)
                                    for vec in reconstructed_soluion._list)
        # prolong in space
        coarse_in_time_fine_in_space = tuple(make_discrete_function(self._d_data[-1]['space'])
                                             for nt in range(len(coarse_solution)))
        for coarse, fine in zip(coarse_solution, coarse_in_time_fine_in_space):
            prolong(coarse, fine)
        # prolong in time
        T = self._T
        coarse_time_grid = OnedGrid(domain=(0., T), num_intervals=(len(self._solution[level]) - 1))
        fine_time_grid = OnedGrid(domain=(0., T), num_intervals=len(self._solution[-1]) - 1)
        self._solution_as_reference[level] = [None for ii in fine_time_grid.centers(1)]
        for n in np.arange(len(fine_time_grid.centers(1))):
            t_n = fine_time_grid.centers(1)[n]
            coarse_entity = min((coarse_time_grid.centers(1) <= t_n).nonzero()[0][-1],
                                coarse_time_grid.size(0) - 1)
            a = coarse_time_grid.centers(1)[coarse_entity]
            b = coarse_time_grid.centers(1)[coarse_entity + 1]
            SF = np.array((1./(a - b)*t_n - b/(a - b), # P1 in tim
                           1./(b - a)*t_n - a/(b - a)))
            U_t = coarse_in_time_fine_in_space[coarse_entity].vector_copy()
            U_t.scal(SF[0][0])
            U_t.axpy(SF[1][0], coarse_in_time_fine_in_space[coarse_entity + 1].vector_copy())
            self._solution_as_reference[level][n] = make_discrete_function(self._d_data[-1]['space'], U_t)

    def _compute_estimates(self, level):
        if not level in self._cache:
            mu = self._d[level].parse_parameter(self.mu)
            eta, (eta_ncs, eta_rs, eta_dfs, time_resiudals, time_derivs_nc) = self._d[level].estimate(
                    self._solution[level], mu)
            self._cache[level] = {
                    'eta_nc': np.linalg.norm(eta_ncs),
                    'eta_df': np.linalg.norm(eta_dfs),
                    'eta_r': np.linalg.norm(eta_rs),
                    'R_T': np.linalg.norm(time_resiudals),
                    'partial_t_nc': np.linalg.norm(time_derivs_nc),
                    'eta': eta}


#!/usr/bin/env python


import numpy as np

from pymor.core.interfaces import BasicInterface


def doerfler_marking(indicators, theta):
    assert 0.0 < theta <= 1.0
    indices = list(range(len(indicators)))
    indicators = [ii**2 for ii in indicators]
    indicators, indices = [list(x) for x in zip(*sorted(zip(indicators, indices),
                                                            key=lambda pair: pair[0],
                                                            reverse=True))]
    total = np.sum(indicators)
    sums = np.array([np.sum(indicators[:ii+1]) for ii in np.arange(len(indicators))])
    where = sums > theta*total
    if np.any(where):
        return indices[:np.argmax(where)+1]
    else:
        return indices


class AdaptiveEnrichment(BasicInterface):

    def __init__(self, grid_and_problem_data, discretization, block_space, reductor, rd,
                 target_error, marking_doerfler_theta, marking_max_age):
        self.grid_and_problem_data = grid_and_problem_data
        self.discretization = discretization
        self.block_space = block_space
        self.reductor = reductor
        self.rd = rd
        self.target_error = target_error
        self.marking_doerfler_theta = marking_doerfler_theta
        self.marking_max_age = marking_max_age

    def _enrich_once(self, U, mu, indicators, age_count):
        marked_subdomains = set(doerfler_marking(indicators, self.marking_doerfler_theta))
        num_dorfler_marked = len(marked_subdomains)
        self.logger.info3('marked {}/{} subdomains due to Dörfler marking'.format(num_dorfler_marked,
            self.block_space.num_blocks))
        for ii in np.where(age_count > self.marking_max_age)[0]:
            marked_subdomains.add(ii)
        num_age_marked = len(marked_subdomains) - num_dorfler_marked
        self.logger.info3('   and {}/{} additionally due to age marking'.format(num_age_marked, self.block_space.num_blocks - num_dorfler_marked))
        self.logger.info3('solving local corrector problems on {} subdomain{} ...'.format(
            len(marked_subdomains), 's' if len(marked_subdomains) > 1 else ''))
        for ii in marked_subdomains:
            self.reductor.enrich_local(ii, U, mu)
        self.rd = self.reductor.reduce()
        # clear age count
        for ii in range(self.block_space.num_blocks):
            if ii in marked_subdomains:
                age_count[ii] = 1
            else:
                age_count[ii] += 1
        return len(marked_subdomains)

    def estimate(self, U, mu, decompose=False):
        return self.rd.estimate(U, mu=mu, decompose=decompose)

    def solve(self, mu, enrichment_steps=np.inf, callback=None):
        mu = self.discretization.parse_parameter(mu)
        enrichment_step = 1
        age_count = np.ones(self.block_space.num_blocks)
        local_problem_solves = 0
        rb_size = self.rd.solution_space.dim
        with self.logger.block('solving {}-dimensional system for mu = {} ...'.format(rb_size, mu)) as _:
            while True:
                U = self.rd.solve(mu)
                eta, _, indicators = self.estimate(U, mu=mu, decompose=True)
                if callback:
                    callback(self.rd, U, mu, {'eta': eta,
                                              'local_problem_solves': local_problem_solves,
                                              'global RB size': self.rd.solution_space.dim,
                                              'local RB sizes': [len(rb) for rb in self.reductor.bases]})
                if eta <= self.target_error:
                    self.logger.info3('estimated error {} below target error of {}, no enrichment required ...'.format(eta, self.target_error))
                    return U, self.rd, self.reductor
                if enrichment_step > enrichment_steps:
                    self.logger.warn('estimated error {} above target error of {}, but stopping since enrichment_steps={} reached!'.format(
                        eta, self.target_error, enrichment_steps))
                    return U, self.rd, self.reductor
                enrichment_step += 1
                self.logger.info3('estimated error {} above target error of {}, enriching ...'.format(eta, self.target_error))
                local_problem_solves = self._enrich_once(U, mu, indicators, age_count)
                self.logger.info3('added {} local basis functions, system size increase: {} --> {}'.format(
                    self.rd.solution_space.dim - rb_size, rb_size, self.rd.solution_space.dim))
                rb_size = self.rd.solution_space.dim

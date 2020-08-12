import pprint

import numpy as np

from pymor.algorithms.system import unblock
from pymor.core.interfaces import ImmutableInterface
from pymor.operators.block import BlockDiagonalOperator
from pymor.operators.constructions import LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.reductors.system import GenericRBSystemReductor
from pymor.parallel.mpi import norm as mpi_norm
from pymor.bindings.dunegdt import DuneGDTVisualizer

from dune.gdt.discretefunction import make_discrete_function
from pymor.vectorarrays.list import ListVectorArray

def _vis(U, global_space, name):
    assert len(U) == 1
    dd = U.data
    ud = make_discrete_function(global_space, name)
    print('vec size {} | space {} | df {}'.format(len(dd[0]), global_space.size(), len(ud)))
    for ii in range(global_space.size()):
        ud.setitem(ii, dd[0][ii])
    ud.visualize(name)

class EstimatorBase(ImmutableInterface):

    def __init__(self, grid, min_diffusion_evs, subdomain_diameters, local_eta_rf_squared, lambda_coeffs, mu_bar, mu_hat,
                 flux_reconstruction, oswald_interpolation_error, mpi_comm, global_rt_space=None,
                 global_dg_space=None):
        self.grid = grid
        self.min_diffusion_evs = min_diffusion_evs
        self.subdomain_diameters = subdomain_diameters
        self.local_eta_rf_squared = local_eta_rf_squared
        self.lambda_coeffs = lambda_coeffs
        self.mu_bar = mu_bar
        self.mu_hat = mu_hat
        self.flux_reconstruction = flux_reconstruction
        self.oswald_interpolation_error = oswald_interpolation_error
        self.num_subdomains = len(grid.subdomains_on_rank)
        self.mpi_comm = mpi_comm
        self.global_rt_space = global_rt_space
        self.global_dg_space = global_dg_space

    def _estimate_elliptic(self, U, mu, d, elliptic_reconstruction=False, decompose=False):
        alpha_mu_mu_bar = self.alpha(self.lambda_coeffs, mu, self.mu_bar)
        gamma_mu_mu_bar = self.gamma(self.lambda_coeffs, mu, self.mu_bar)
        alpha_mu_mu_hat = self.alpha(self.lambda_coeffs, mu, self.mu_hat)

        vec_size = self.num_subdomains
        local_eta_nc = np.zeros((vec_size, len(U)))
        local_eta_r = np.zeros((vec_size, len(U)))
        local_eta_df = np.zeros((vec_size, len(U)))

        U_r = self.flux_reconstruction.apply(U, mu=mu)
        U_r_u = self.flux_reconstruction.operators[0].range.from_data(U_r.data)
        # _vis(U_r, self.global_rt_space, name=f'flux_recon_{mu["diffusion"][0]}')

        U_o = self.oswald_interpolation_error.apply(U)
        U_o_u = self.oswald_interpolation_error.range.from_data(U_o.data)
        # _vis(U_o, self.global_dg_space, name=f'oswald_inter_error_{mu["diffusion"][0]}')

        if elliptic_reconstruction:
            assert False
            BU = d.operator.apply(U, mu=mu)
            BU_R = d.l2_product.apply_inverse(BU)
            F_R = d.l2_product.apply_inverse(d.rhs.as_source_array())
            BUF_R = BU_R - F_R

        for ii, subdomain in enumerate(self.grid.subdomains_on_rank):
            local_eta_nc[ii] = d.operators['nc_{}'.format(subdomain)].pairwise_apply2(U_o, U_o, mu=mu)
            local_eta_r[ii] += self.local_eta_rf_squared[ii]
            r_fd = d.operators['r_fd_{}'.format(subdomain)].apply(U_r, mu=mu).data[:, 0]
            local_eta_r[ii] -= 2*r_fd
            r_dd = d.operators['r_dd_{}'.format(subdomain)].pairwise_apply2(U_r, U_r, mu=mu)
            local_eta_r[ii] += r_dd
            self.logger.debug('Subdomain {}: r_fd {} | r_dd {}'.format(subdomain, r_fd, r_dd))
            if elliptic_reconstruction:
                local_eta_r[ii] += d.operators['r_l2_{}'.format(subdomain)].pairwise_apply2(BU_R, BU_R)
                local_eta_r[ii] -= d.operators['r_l2_{}'.format(subdomain)].pairwise_apply2(F_R, F_R)
                local_eta_r[ii] -= 2*d.operators['r_ud_{}'.format(subdomain)].pairwise_apply2(BUF_R, U_r, mu=mu)

            local_eta_df[ii] += d.operators['df_aa_{}'.format(subdomain)].pairwise_apply2(U, U, mu=mu)
            local_eta_df[ii] += d.operators['df_bb_{}'.format(subdomain)].pairwise_apply2(U_r, U_r, mu=mu)
            local_eta_df[ii] += 2*d.operators['df_ab_{}'.format(subdomain)].pairwise_apply2(U, U_r, mu=mu)

            # eta r, scale
            poincaree_constant = 1./(np.pi**2)
            min_diffusion_ev = self.min_diffusion_evs[ii]
            subdomain_h = self.subdomain_diameters[ii]
            local_eta_r[ii] *= (poincaree_constant/min_diffusion_ev) * subdomain_h**2

        with np.printoptions(precision=6, suppress=True):
            self.logger.debug('Estimated for subdomains {}'.format(self.grid.subdomains_on_rank))
            err = {'eta_nc': local_eta_nc, 'eta_r': local_eta_r, 'eta_df': local_eta_df}
            self.logger.debug('\n'+pprint.pformat(err))
            # self.logger.debug()

        eta = 0.
        eta += np.sqrt(gamma_mu_mu_bar)      * mpi_norm(local_eta_nc)
        eta += (1./np.sqrt(alpha_mu_mu_hat)) * mpi_norm(local_eta_r + local_eta_df)
        eta *= 1./np.sqrt(alpha_mu_mu_bar)

        if decompose:
            local_indicators = np.array(
                [(2./alpha_mu_mu_bar) * (gamma_mu_mu_bar * local_eta_nc[ii]**2 +
                                         (1./alpha_mu_mu_hat) * (local_eta_r[ii] + local_eta_df[ii])**2)
                 for ii in range(self.num_subdomains)]
            )
            return eta, (local_eta_nc, local_eta_r, local_eta_df), local_indicators
        else:
            return eta

    def alpha(self, thetas, mu, mu_bar):
        result = np.inf
        for theta in thetas:
            theta_mu = theta.evaluate(mu)
            theta_mu_bar = theta.evaluate(mu_bar)
            assert theta_mu/theta_mu_bar > 0
            result = np.min((result, theta_mu/theta_mu_bar))
            return result

    def gamma(self, thetas, mu, mu_bar):
        result = -np.inf
        for theta in thetas:
            theta_mu = theta.evaluate(mu)
            theta_mu_bar = theta.evaluate(mu_bar)
            assert theta_mu/theta_mu_bar > 0
            result = np.max((result, theta_mu/theta_mu_bar))
        return result


class EllipticEstimator(EstimatorBase):

    def estimate(self, U, mu, d, decompose=False):
        return self._estimate_elliptic(U, mu, d, False, decompose)


class ParabolicEstimator(EstimatorBase):

    def estimate(self, U, mu, d, decompose=False):
        dt = d.T / d.time_stepper.nt

        eta, (local_eta_nc, local_eta_r, local_eta_df), elliptic_local_indicators = \
            self._estimate_elliptic(U, mu, d, True, True)

        # time_residual = self.residual_operator.apply(U[1:] - U[:-1], mu)
        time_residual = d.operator.apply(U[1:] - U[:-1], mu)
        time_residual = d.l2_product.apply_inverse(time_residual).pairwise_dot(time_residual)
        time_residual *= dt / 3
        time_residual = np.sqrt(time_residual)

        # elliptic error
        eta *= 2 * np.sqrt(dt / 3)
        local_eta_nc *= 2 * np.sqrt(dt / 3)
        local_eta_r *= 2 * np.sqrt(dt / 3)
        local_eta_df *= 2 * np.sqrt(dt / 3)

        U_o = self.oswald_interpolation_error.apply(U)
        U_o_diff = U_o[1:] - U_o[:-1]
        time_deriv_nc = np.zeros((self.num_subdomains, len(U) - 1))
        for ii in range(self.num_subdomains):
            time_deriv_nc[ii] = d.operators['nc_{}'.format(ii)].pairwise_apply2(U_o_diff, U_o_diff, mu=mu)
        time_deriv_nc *= 1 / dt
        time_deriv_nc = np.sqrt(time_deriv_nc)

        est = np.linalg.norm(eta) + np.linalg.norm(time_residual) + np.linalg.norm(time_deriv_nc)
        return est, (local_eta_nc, local_eta_r, local_eta_df, time_residual, time_deriv_nc)

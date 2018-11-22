import numpy as np

from pymor.algorithms.system import unblock
from pymor.core.interfaces import ImmutableInterface
from pymor.operators.block import BlockDiagonalOperator
from pymor.operators.constructions import LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.reductors.system import GenericRBSystemReductor


class LRBMSReductor(GenericRBSystemReductor):

    def __init__(self, d, bases=None, products=None, order=None, num_cpus=1):
        assert order is None or 0 <= order <= 1
        super().__init__(d, bases=bases, products=products)

        if order is None and bases is None:
            order = 0
        if order is not None:
            self.logger.info(
                'initializing local reduced bases with DG shape functions of up to order {} ... '.format(order)
            )
            for ii in range(len(d.solution_space.subspaces)):
                self.extend_basis_local(d.shape_functions(ii, order))

    def _reduce(self):
        d = self.d

        self.logger.info('Computing oswald interpolations ...')
        oi = d.estimator.oswald_interpolation_error

        oi_red = []
        for i, OI_i_space in enumerate(oi.range.subspaces):
            oi_i = oi._blocks[i, i]
            basis = self.bases[oi_i.source.id]
            self.bases[OI_i_space.id] = oi_i.apply(basis)
            oi_red.append(NumpyMatrixOperator(np.eye(len(basis)),
                                              source_id=oi_i.source.id, range_id=oi_i.range.id))
        oi_red = unblock(BlockDiagonalOperator(oi_red))

        self.logger.info('Computing flux reconstructions ...')
        fr = d.estimator.flux_reconstruction

        for i, RT_i_space in enumerate(fr.range.subspaces):
            self.bases[RT_i_space.id] = RT_i_space.empty()

        red_aff_components = []
        for i_aff, aff_component in enumerate(fr.operators):
            red_aff_component = []
            for i, RT_i_space in enumerate(aff_component.range.subspaces):
                fr_i = aff_component._blocks[i, i]
                basis = self.bases[fr_i.source.id]
                self.bases[RT_i_space.id].append(fr_i.apply(basis))
                M = np.zeros((len(basis) * len(fr.operators), len(basis)))
                M[i_aff * len(basis): (i_aff+1) * len(basis), :] = np.eye(len(basis))
                red_aff_component.append(NumpyMatrixOperator(M, source_id=fr_i.source.id, range_id=fr_i.range.id))
            red_aff_components.append(BlockDiagonalOperator(red_aff_component))
        fr_red = LincombOperator(red_aff_components, fr.coefficients)
        fr_red = unblock(fr_red)

        red_estimator = d.estimator.with_(flux_reconstruction=fr_red, oswald_interpolation_error=oi_red)

        rd = super()._reduce()
        rd = rd.with_(estimator=red_estimator)

        return rd

    def enrich_local(self, subdomain, U, mu=None):
        Us = [self.reconstruct_local(U, 'domain_{}'.format(sdi)) for sdi in self.d.neighborhoods[subdomain]]
        local_correction = self.d.solve_for_local_correction(subdomain, Us, mu)
        self.extend_basis_local(local_correction)


class EstimatorBase(ImmutableInterface):

    def __init__(self, grid, min_diffusion_evs, subdomain_diameters, local_eta_rf_squared, lambda_coeffs, mu_bar, mu_hat,
                 flux_reconstruction, oswald_interpolation_error, mpi_comm):
        self.grid = grid
        self.min_diffusion_evs = min_diffusion_evs
        self.subdomain_diameters = subdomain_diameters
        self.local_eta_rf_squared = local_eta_rf_squared
        self.lambda_coeffs = lambda_coeffs
        self.mu_bar = mu_bar
        self.mu_hat = mu_hat
        self.flux_reconstruction = flux_reconstruction
        self.oswald_interpolation_error = oswald_interpolation_error
        self.num_subdomains = len(subdomain_diameters)

    def _estimate_elliptic(self, U, mu, d, elliptic_reconstruction=False, decompose=False):
        alpha_mu_mu_bar = self.alpha(self.lambda_coeffs, mu, self.mu_bar)
        gamma_mu_mu_bar = self.gamma(self.lambda_coeffs, mu, self.mu_bar)
        alpha_mu_mu_hat = self.alpha(self.lambda_coeffs, mu, self.mu_hat)

        local_eta_nc = np.zeros((self.num_subdomains, len(U)))
        local_eta_r = np.zeros((self.num_subdomains, len(U)))
        local_eta_df = np.zeros((self.num_subdomains, len(U)))

        U_r = self.flux_reconstruction.apply(U, mu=mu)
        U_o = self.oswald_interpolation_error.apply(U)

        if elliptic_reconstruction:
            BU = d.operator.apply(U, mu=mu)
            BU_R = d.l2_product.apply_inverse(BU)
            F_R = d.l2_product.apply_inverse(d.rhs.as_source_array())
            BUF_R = BU_R - F_R

        for ii in range(self.num_subdomains):
            local_eta_nc[ii] = d.operators['nc_{}'.format(ii)].pairwise_apply2(U_o, U_o, mu=mu)
            local_eta_r[ii] += self.local_eta_rf_squared[ii]
            local_eta_r[ii] -= 2*d.operators['r_fd_{}'.format(ii)].apply(U_r, mu=mu).data[:, 0]
            local_eta_r[ii] += d.operators['r_dd_{}'.format(ii)].pairwise_apply2(U_r, U_r, mu=mu)
            if elliptic_reconstruction:
                local_eta_r[ii] += d.operators['r_l2_{}'.format(ii)].pairwise_apply2(BU_R, BU_R)
                local_eta_r[ii] -= d.operators['r_l2_{}'.format(ii)].pairwise_apply2(F_R, F_R)
                local_eta_r[ii] -= 2*d.operators['r_ud_{}'.format(ii)].pairwise_apply2(BUF_R, U_r, mu=mu)

            local_eta_df[ii] += d.operators['df_aa_{}'.format(ii)].pairwise_apply2(U, U, mu=mu)
            local_eta_df[ii] += d.operators['df_bb_{}'.format(ii)].pairwise_apply2(U_r, U_r, mu=mu)
            local_eta_df[ii] += 2*d.operators['df_ab_{}'.format(ii)].pairwise_apply2(U, U_r, mu=mu)

            # eta r, scale
            poincaree_constant = 1./(np.pi**2)
            min_diffusion_ev = self.min_diffusion_evs[ii]
            subdomain_h = self.subdomain_diameters[ii]
            local_eta_r[ii] *= (poincaree_constant/min_diffusion_ev) * subdomain_h**2

        local_eta_nc = np.sqrt(local_eta_nc)
        local_eta_r = np.sqrt(local_eta_r)
        local_eta_df = np.sqrt(local_eta_df)

        eta = 0.
        eta += np.sqrt(gamma_mu_mu_bar)      * np.linalg.norm(local_eta_nc, axis=0)
        eta += (1./np.sqrt(alpha_mu_mu_hat)) * np.linalg.norm(local_eta_r + local_eta_df, axis=0)
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


class ParabolicLRBMSReductor(LRBMSReductor):

    pass

    # def _reduce(self):
    #     d = self.d

    #     if not isinstance(d.operator, LincombOperator) and all(isinstance(op, BlockOperator) for op in
    #                                                            d.operator.operators):
    #         raise NotImplementedError

    #     residual_source_bases = [self.bases[ss.id] for ss in d.operator.source.subspaces]
    #     residual_range_bases = []
    #     for ii in range(len(residual_source_bases)):
    #         b = residual_source_bases[ii].empty()
    #         for op in d.operator.operators:
    #             for o in op._blocks[ii, :]:
    #                 if not isinstance(o, ZeroOperator):
    #                     b.append(o.apply(self.bases[o.source.id]))
    #         p = d.l2_product._blocks[ii, ii]
    #         b = p.apply_inverse(b)
    #         gram_schmidt(b, product=p, copy=False)
    #         residual_range_bases.append(b)

    #     residual_operator = project_system(d.operator, residual_range_bases, residual_source_bases)
    #     residual_operator = unblock(residual_operator)

    #     rd = super()._reduce()
    #     rd = rd.with_(estimator=rd.estimator.with_(residual_operator=residual_operator,
    #                                                residual_product=None))

    #     return rd


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

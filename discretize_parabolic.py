import numpy as np

from pymor.algorithms.timestepping import ImplicitEulerTimeStepper
from pymor.bindings.dunext import DuneXTMatrixOperator
from pymor.bindings.dunegdt import DuneGDTVisualizer
from pymor.discretizations.basic import InstationaryDiscretization

from discretize_elliptic import DuneDiscretizationBase, EllipticEstimator
from discretize_elliptic import discretize as discretize_ell


class InstationaryDuneDiscretization(DuneDiscretizationBase, InstationaryDiscretization):

    def __init__(self, global_operator, global_rhs, global_mass,
                 T, initial_data, operator, rhs, mass=None, time_stepper=None, num_values=None,
                 products=None, operators=None, parameter_space=None, estimator=None, visualizer=None,
                 cache_region=None, name=None):
        super().__init__(T, initial_data, operator, rhs, mass=mass, time_stepper=time_stepper, num_values=num_values,
                         products=products, operators=operators, parameter_space=parameter_space, estimator=estimator,
                         visualizer=visualizer, cache_region=cache_region, name=name)
        self.global_operator, self.global_rhs, self.global_mass = \
            global_operator, global_rhs, global_mass

    def _solve(self, mu):
        mu = self.parse_parameter(mu).copy()

        # explicitly checking if logging is disabled saves the expensive str(mu) call
        if not self.logging_disabled:
            self.logger.info('Solving {} for {} ...'.format(self.name, mu))

        mu['_t'] = 0
        U0 = self.initial_data.as_range_array(mu)
        U = self.time_stepper.solve(operator=self.global_operator, rhs=self.global_rhs,
                                    initial_data=self.unblock(U0), mass=self.global_mass,
                                    initial_time=0, end_time=self.T, mu=mu, num_values=self.num_values)

        return self.solution_space.from_data(U.data)


class ParabolicEstimator(EllipticEstimator):

    def __init__(self, residual_operator, residual_product,
                 min_diffusion_evs, subdomain_diameters, local_eta_rf_squared, lambda_coeffs, mu_bar, mu_hat,
                 flux_reconstruction, oswald_interpolation_error):
        super().__init__(min_diffusion_evs, subdomain_diameters, local_eta_rf_squared, lambda_coeffs, mu_bar, mu_hat,
                         flux_reconstruction, oswald_interpolation_error)
        self.residual_operator = residual_operator
        self.residual_product = residual_product

    def estimate(self, U, mu, discretization, decompose=False):
        d = discretization
        dt = d.T / d.time_stepper.nt

        time_residual = self.residual_operator.apply(U[1:] - U[:-1], mu)
        if self.residual_product:
            time_residual = self.residual_product.apply_inverse(time_residual).pairwise_dot(time_residual)
        else:
            time_residual = time_residual.l2_norm2()
        time_residual *= dt / 3
        time_residual = np.sqrt(time_residual)

        return time_residual


def discretize(grid_and_problem_data, T, nt):
    d, d_data = discretize_ell(grid_and_problem_data)
    block_space = d_data['block_space']
    # assemble global L2 product
    l2_mat = d.global_operator.operators[0].matrix.copy()  # to ensure matching pattern
    l2_mat.scal(0.)
    for ii in range(block_space.num_blocks):
        local_l2_product = d.l2_product._blocks[ii, ii]
        block_space.mapper.copy_local_to_global(local_l2_product.matrix,
                                                local_l2_product.matrix.pattern(),
                                                ii,
                                                l2_mat)
    mass = d.l2_product
    operators = {k: v for k, v in d.operators.items() if k not in d.special_operators}
    global_mass = DuneXTMatrixOperator(l2_mat)

    e = d.estimator
    estimator = ParabolicEstimator(d.operator,
                                   d.l2_product,
                                   e.min_diffusion_evs, e.subdomain_diameters, e.local_eta_rf_squared, e.lambda_coeffs,
                                   e.mu_bar, e.mu_hat, e.flux_reconstruction, e.oswald_interpolation_error)

    d = InstationaryDuneDiscretization(d.global_operator,
                                       d.global_rhs,
                                       global_mass,
                                       T,
                                       d.operator.source.zeros(1),
                                       d.operator,
                                       d.rhs,
                                       mass=mass,
                                       time_stepper=ImplicitEulerTimeStepper(nt=nt,
                                                                             solver_options='operator'),
                                       products=d.products,
                                       operators=operators,
                                       estimator=estimator,
                                       parameter_space=d.parameter_space,
                                       visualizer=DuneGDTVisualizer(block_space))

    return d, d_data

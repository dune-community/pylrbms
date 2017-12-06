import numpy as np

from pymor.algorithms.timestepping import ImplicitEulerTimeStepper
from pymor.bindings.dunext import DuneXTMatrixOperator
from pymor.bindings.dunegdt import DuneGDTVisualizer
from pymor.discretizations.basic import InstationaryDiscretization
from pymor.operators.constructions import Concatenation
from pymor.parameters.spaces import CubicParameterSpace

from discretize_elliptic_block_swipdg import DuneDiscretizationBase
from discretize_elliptic_block_swipdg import discretize as discretize_ell
from lrbms import ParabolicEstimator


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


def discretize(grid_and_problem_data, T, nt):
    d, d_data = discretize_ell(grid_and_problem_data)
    assert isinstance(d.parameter_space, CubicParameterSpace)
    parameter_range = grid_and_problem_data['parameter_range']
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

    local_div_ops, local_l2_products, local_projections, local_rt_projections = \
        d_data['local_div_ops'], d_data['local_l2_products'], d_data['local_projections'], d_data['local_rt_projections']

    for ii in range(d_data['grid'].num_subdomains):

        local_div = Concatenation([local_div_ops[ii], local_rt_projections[ii]])

        operators['r_ud_{}'.format(ii)] = \
            Concatenation([local_projections[ii].T, local_l2_products[ii], local_div], name='r_ud_{}'.format(ii))

        operators['r_l2_{}'.format(ii)] = \
            Concatenation([local_projections[ii].T, local_l2_products[ii], local_projections[ii]],
                          name='r_l2_{}'.format(ii))

    e = d.estimator
    estimator = ParabolicEstimator(e.min_diffusion_evs, e.subdomain_diameters, e.local_eta_rf_squared, e.lambda_coeffs,
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
                                       visualizer=DuneGDTVisualizer(block_space))
    d = d.with_(parameter_space=CubicParameterSpace(d.parameter_type, parameter_range[0], parameter_range[1]))

    return d, d_data

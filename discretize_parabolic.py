from dune.xt.la import IstlRowMajorSparseMatrixDouble as Matrix
from dune.gdt import (
        make_l2_matrix_operator,
        make_system_assembler
)

from pymor.algorithms.timestepping import ImplicitEulerTimeStepper
from pymor.bindings.dunext import DuneXTMatrixOperator
from pymor.bindings.dunegdt import DuneGDTVisualizer
from pymor.discretizations.basic import InstationaryDiscretization

from discretize_elliptic import *
from discretize_elliptic import discretize as discretize_ell


class InstationaryDuneDiscretization(InstationaryDiscretization):

    def __init__(self, T, initial_data, operator, rhs, mass=None, time_stepper=None, num_values=None,
                 products=None, operators=None, parameter_space=None, estimator=None, visualizer=None,
                 cache_region=None, name=None):
        super().__init__(T, initial_data, operator, rhs, mass=mass, time_stepper=time_stepper, num_values=num_values,
                         products=products, operators=operators, parameter_space=parameter_space, estimator=estimator,
                         visualizer=visualizer, cache_region=cache_region, name=name)

    def _solve(self, mu):
        mu = self.parse_parameter(mu).copy()

        # explicitly checking if logging is disabled saves the expensive str(mu) call
        if not self.logging_disabled:
            self.logger.info('Solving {} for {} ...'.format(self.name, mu))

        mu['_t'] = 0
        U0 = self.initial_data.as_range_array(mu)
        U = self.time_stepper.solve(operator=self.operators['global_op'], rhs=self.operators['global_rhs'],
                                    initial_data=self.unblock(U0), mass=self.operators['global_mass'],
                                    initial_time=0, end_time=self.T, mu=mu, num_values=self.num_values)

        return self.solution_space.from_data(U.data)

    def visualize(self, U, *args, **kwargs):
        self.visualizer.visualize(self.unblock(U), self, *args, **kwargs)

    def unblock(self, U):
        return self.operators['global_op'].source.from_data(U.data)

    def as_generic_type(self):
        ops = dict(self.operators)
        for op in ('operator',
                   'rhs',
                   'initial_data',
                   'mass',
                   'global_op',
                   'global_rhs',
                   'global_mass'):
            if op in ops.keys():
                del ops[op]

        return InstationaryDiscretization(self.T, self.initial_data, self.operator, self.rhs, self.mass, self.time_stepper,
                                          self.num_values, products=self.products, operators=ops,
                                          parameter_space=self.parameter_space)

    def shape_functions(self, subdomain, order=0):
        assert 0 <= order <= 1
        local_space = self.solution_space.subspaces[subdomain]
        U = local_space.make_array([Vector(local_space.dim, 1.)])

        if order == 1:
            from dune.gdt import make_discrete_function, project
            dune_local_space = self.visualizer.space.local_space(subdomain)
            tmp_discrete_function = make_discrete_function(dune_local_space)
            for expression in ('x[0]', 'x[1]', 'x[0]*x[1]'):
                func = make_expression_function_1x1(grid, 'x', expression, order=2)
                project(func, tmp_discrete_function)
                U.append(local_space.make_array([tmp_discrete_function.vector_copy()]))

        return U



def discretize(grid_and_problem_data, T, nt):
    d, block_space = discretize_ell(grid_and_problem_data)
    # assemble global L2 product
    l2_mat = d.operators['global_op'].operators[0].matrix.copy() # to ensure matching pattern
    l2_mat.scal(0.)
    for ii in range(block_space.num_blocks):
        local_l2_product = d.operators['local_l2_product_{}'.format(ii)]
        block_space.mapper.copy_local_to_global(local_l2_product.matrix,
                                                local_l2_product.matrix.pattern(),
                                                ii,
                                                l2_mat)
    mass = BlockDiagonalOperator([d.operators['local_l2_product_{}'.format(ii)] for ii in range(block_space.num_blocks)])
    ops = {k: v for k, v in d.operators.items()
           if k != 'operator' and k != 'rhs' and k != 'mass'}
    ops['global_mass'] = DuneXTMatrixOperator(l2_mat)
    d = InstationaryDuneDiscretization(T,
                                       d.operator.source.zeros(1),
                                       d.operator,
                                       d.rhs,
                                       mass=mass,
                                       time_stepper=ImplicitEulerTimeStepper(nt=nt,
                                                                             solver_options='operator'),
                                       products=d.products,
                                       operators=ops,
                                       estimator=d.estimator,
                                       parameter_space=d.parameter_space,
                                       visualizer=DuneGDTVisualizer(block_space))

    return d, block_space

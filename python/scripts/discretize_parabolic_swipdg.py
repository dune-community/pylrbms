from pymor.algorithms.timestepping import ImplicitEulerTimeStepper
from pymor.bindings.dunegdt import DuneGDTVisualizer
from pymor.discretizations.basic import InstationaryDiscretization
from pymor.parameters.spaces import CubicParameterSpace

from discretize_elliptic_swipdg import discretize as discretize_stationary


def discretize(grid_and_problem_data, T, nt, polorder):

    d, data = discretize_stationary(grid_and_problem_data, polorder)
    assert isinstance(d.parameter_space, CubicParameterSpace)
    parameter_range = grid_and_problem_data['parameter_range']

    d = InstationaryDiscretization(T,
                                   d.operator.source.zeros(1),
                                   d.operator,
                                   d.rhs,
                                   mass=d.operators['l2'],
                                   time_stepper=ImplicitEulerTimeStepper(nt=nt,
                                                                         solver_options='operator'),
                                   products=d.products,
                                   operators={kk: vv for kk, vv in d.operators.items()
                                              if kk not in ('operator', 'rhs') and vv not in (d.operators, d.rhs)},
                                   visualizer=DuneGDTVisualizer(data['space']))
    d = d.with_(parameter_space=CubicParameterSpace(d.parameter_type, parameter_range[0], parameter_range[1]))

    return d, data


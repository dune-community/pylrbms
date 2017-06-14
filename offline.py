#!/usr/bin/env python

from dune.xt.la import IstlDenseVectorDouble as Vector
from dune.gdt import make_discrete_function, project

from pymor.bindings.dunext import DuneXTVector
from pymor.core.logger import getLogger
from pymor.reductors.system import GenericRBSystemReductor
from pymor.vectorarrays.list import ListVectorArray


def init_local_reduced_bases(d, block_space, order):
    logger = getLogger('offline.init_local_reduced_bases')
    if order > 1:
        order = 1
    U = d.solution_space.empty()
    reductor = GenericRBSystemReductor(d, products=[d.operators['local_energy_dg_product_{}'.format(ii)]
                                                    for ii in range(block_space.num_blocks)])
    if order >= 0:
        logger.info('initializing local reduced bases with DG shape functions of up to order {} ... '.format(order))

        # order 0 basis
        for ii in range(block_space.num_blocks):
            local_space = block_space.local_space(ii)
            reductor.extend_basis_local(ListVectorArray([DuneXTVector(Vector(local_space.size(), 1.)), ],
                                                        U._blocks[ii].space))
    if order >= 1:
        for ii in range(block_space.num_blocks):
            local_space = block_space.local_space(ii)
            tmp_discrete_function = make_discrete_function(local_space)
            for expression in ('x[0]', 'x[1]', 'x[0]*x[1]'):
                func = make_expression_function_1x1(grid, 'x', expression, order=2)
                project(func, tmp_discrete_function)
                reductor.extend_basis_local(ListVectorArray([DuneXTVector(tmp_discrete_function.vector_copy()), ],
                                                            U._blocks[ii].space))
            del tmp_discrete_function

    return reductor


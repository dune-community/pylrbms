import itertools
import pprint

import numpy as np
from mpi4py import MPI

from pymor.algorithms.system import unblock
from pymor.core.interfaces import ImmutableInterface
from pymor.operators.block import BlockDiagonalOperator
from pymor.vectorarrays.numpy import NumpyVectorArray
from pymor.operators.constructions import LincombOperator, VectorArrayOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.reductors.system import GenericRBSystemReductor
from pymor.parallel.mpi import norm as mpi_norm


class LRBMSReductor(GenericRBSystemReductor):

    def __init__(self, d, bases=None, products=None, order=None, num_cpus=1, solver_options=None):
        assert order is None or 0 <= order <= 1
        self.solver_options = solver_options
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
        local_correction = self.d.solve_for_local_correction(subdomain, Us, mu, inverse_options=self.solver_options)
        self.extend_basis_local(local_correction)


class ParallelLRBMSReductor(LRBMSReductor):

    def __init__(self, d, bases=None, products=None, order=None, solver_options=None, mpi_comm=MPI.COMM_WORLD):
        super().__init__(d, bases=bases, products=products, solver_options=solver_options, num_cpus=1, order=order)
        self.mpi_comm = mpi_comm

    def _op_sum(self, op, name='None'):
        self.logger.debug('REDUCE SUM op {}'.format(op.name))
        if isinstance(op, NumpyMatrixOperator):
            orig_type = type(op.matrix)
            mat = op.matrix.todense() if op.sparse else op.matrix
            target = np.zeros_like(mat)
            self.mpi_comm.Allreduce([mat.data, MPI.DOUBLE], [target, MPI.DOUBLE], MPI.SUM)
            if orig_type == np.ndarray:
                return op.with_(matrix=target)
            else:
                new_mat = orig_type(mat)
                return op.with_(matrix=new_mat)
        elif isinstance(op, LincombOperator):
            new_ops = [self._op_sum(l_op, 'Lincomb {}_{}'.format(name, ii)) for ii,l_op in enumerate(op.operators)]
            return op.with_(operators=new_ops)
        elif isinstance(op, VectorArrayOperator):
            # return op
            arr = op._array
            data = arr.data
            assert isinstance(data, np.ndarray)
            target = np.zeros_like(data)
            self.logger.error('VEC REDUCE {} -- {}'.format(data.shape, arr.dim))
            # self.mpi_comm.barrier()
            other_shape = self.mpi_comm.allgather(data.shape)
            self.logger.debug('REDUCE {} shapes {} -- {}'.format(name, other_shape, data.shape))
            other_data = np.zeros_like(data)
            self.logger.debug(pprint.pformat(data))
            other_data = self.mpi_comm.allreduce(data, MPI.SUM)
            data = other_data
            assert data.shape
            # self.mpi_comm.Allreduce([data, MPI.DOUBLE], [target, MPI.DOUBLE], MPI.SUM)
            return op.with_(array=NumpyVectorArray(data, arr.space))
        else:
            self.logger.error('OTher'+op.name)
            raise RuntimeError()

    def _reduce(self):
        rd = super()._reduce()
        return rd
        whitelist = ['rhs', 'operator']
        self.logger.debug('REDUCE OPS {}'.format(','.join((n for n,o in rd.operators.items() if n in whitelist))))
        self.logger.debug('UNREDUCE OPS {}'.format(','.join((n for n,o in rd.operators.items() if n not in whitelist))))
        ops = {}
        for n,o in rd.operators.items():
            if n in whitelist:
                # self.mpi_comm.barrier()
                ops[n] = self._op_sum(o)
            else:
                ops[n] = o
        prods = {}
        for n,o in rd.products.items():

            self.mpi_comm.barrier()
            ops[n] = self._op_sum(o)

        rd = rd.with_(operators=ops, products=prods)
        return rd

    def enrich_local(self, subdomain, U, mu=None):
        super().enrich_local(subdomain=subdomain, U=U, mu=mu)


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

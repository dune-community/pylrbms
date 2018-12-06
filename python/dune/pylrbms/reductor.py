import pprint

import numpy as np

from pymor.algorithms.system import unblock
from pymor.core.interfaces import ImmutableInterface
from pymor.operators.block import BlockDiagonalOperator
from pymor.operators.constructions import LincombOperator
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
        local_correction = self.d.solve_for_local_correction(subdomain, Us, mu)
        self.extend_basis_local(local_correction)


class ParallelLRBMSReductor(GenericRBSystemReductor):

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
        local_correction = self.d.solve_for_local_correction(subdomain, Us, mu)
        self.extend_basis_local(local_correction)


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

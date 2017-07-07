import numpy as np

from pymor.reductors.system import GenericRBSystemReductor
from pymor.operators.block import BlockDiagonalOperator
from pymor.operators.constructions import LincombOperator
from pymor.algorithms.system import unblock
from pymor.operators.numpy import NumpyMatrixOperator


class LRBMSReductor(GenericRBSystemReductor):

    def __init__(self, d, bases=None, products=None, order=None):
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

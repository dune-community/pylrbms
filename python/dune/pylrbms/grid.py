#!/usr/bin/env python
import dune.xt.grid.provider
import dune.gdt
from pymor.core.logger import getLogger
from mpi4py import MPI


def make_grid(domain=([0, 0], [1, 1]),
              num_subdomains=None,
              half_num_fine_elements_per_subdomain_and_dim=3,
              inner_boundary_segment_index=18446744073709551573,
              mpi_comm=MPI.COMM_WORLD):
    logger = getLogger('grid.grid')
    logger.info('initializing grid ... ')

    if num_subdomains:
        nm = 'make_cube_dd_subdomains_grid__{}'.format(dune.gdt.GDT_BINDINGS_GRID)
        _make_grid = getattr(dune.xt.grid.provider, nm)
        return _make_grid(
                lower_left=domain[0],
                upper_right=domain[1],
                num_elements=[num_subdomains[0]*half_num_fine_elements_per_subdomain_and_dim,
                              num_subdomains[1]*half_num_fine_elements_per_subdomain_and_dim],
                num_refinements=2,
                num_partitions=num_subdomains,
                num_oversampling_layers=4*half_num_fine_elements_per_subdomain_and_dim,
                inner_boundary_segment_index=inner_boundary_segment_index,
        mpi_comm=mpi_comm)

    nm = 'make_cube_grid__{}'.format(dune.gdt.GDT_BINDINGS_GRID)
    _make_grid = getattr(dune.xt.grid.provider, nm)

    return _make_grid(
            lower_left=domain[0],
            upper_right=domain[1],
            num_elements=[half_num_fine_elements_per_subdomain_and_dim,
                          half_num_fine_elements_per_subdomain_and_dim],
            num_refinements=2,
    mpi_comm=mpi_comm)


def make_boundary_info(grid, config):
    from dune.xt.grid.boundaryinfo import (
        make_boundary_info_on_dd_subdomain_boundary_layer,
        make_boundary_info_on_leaf_layer
    )
    try:
        return make_boundary_info_on_dd_subdomain_boundary_layer(grid, config)
    except:
        return make_boundary_info_on_leaf_layer(grid, config)


def grid_info(log, grid):
    tpl = '''
**************************************************************
* Grid Type {}
* # Subdomains {}
* Process subdomains {}
* First Neighbors {}
* Boundary Subdomains {}
**************************************************************
    '''
    log(tpl.format(str(type(grid)), grid.num_subdomains, grid.subdomains_on_rank,
        grid.neighboring_subdomains(grid.subdomains_on_rank[0]), grid.boundary_subdomains()))

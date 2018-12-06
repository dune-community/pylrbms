#!/usr/bin/env python

from pymor.core.logger import getLogger
from mpi4py import MPI


def make_grid(domain=([0, 0], [1, 1]),
              num_subdomains=None,
              half_num_fine_elements_per_subdomain_and_dim=3,
              inner_boundary_segment_index=18446744073709551573,
              grid_type = 'yasp',
              mpi_comm=MPI.COMM_WORLD):
    logger = getLogger('grid.grid')
    logger.info('initializing grid ... ')

    if num_subdomains:
        if grid_type == 'yasp':
            from dune.xt.grid.provider import make_cube_dd_subdomains_grid__2d_cube_yaspgrid as _make_grid
        else:
            from dune.xt.grid.provider import make_cube_dd_subdomains_grid__2d_simplex_alunonconformgrid as _make_grid

        return _make_grid(
                lower_left=domain[0],
                upper_right=domain[1],
                num_elements=[num_subdomains[0]*half_num_fine_elements_per_subdomain_and_dim,
                              num_subdomains[1]*half_num_fine_elements_per_subdomain_and_dim],
                num_refinements=2,
                num_partitions=num_subdomains,
                num_oversampling_layers=2*half_num_fine_elements_per_subdomain_and_dim,
                inner_boundary_segment_index=inner_boundary_segment_index,
        mpi_comm=mpi_comm)

    if grid_type == 'yasp':
        from dune.xt.grid.provider import make_cube_grid__2d_cube_yaspgrid as _make_grid
    else:
        from dune.xt.grid.provider import make_cube_grid__2d_simplex_aluconformgrid as _make_grid

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

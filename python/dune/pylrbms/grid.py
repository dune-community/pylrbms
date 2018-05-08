#!/usr/bin/env python

from dune.xt.common import init_logger, init_mpi
try:
    init_mpi()
except:
    pass
# init_logger()

from pymor.core.logger import getLogger


def make_grid(domain=([0, 0], [1, 1]),
              num_subdomains=[3, 4],
              half_num_fine_elements_per_subdomain_and_dim=3,
              inner_boundary_segment_index=18446744073709551573,
              grid_type = 'yasp'):
    logger = getLogger('grid.grid')
    logger.info('initializing grid ... ')

    if grid_type == 'yasp':
        from dune.xt.grid import make_cube_dd_subdomains_grid__2d_cube_yaspgrid as _make_grid
    else:
        from dune.xt.grid import make_cube_dd_subdomains_grid__2d_simplex_aluconformgrid as _make_grid

    return _make_grid(
            lower_left=domain[0],
            upper_right=domain[1],
            num_elements=[num_subdomains[0]*half_num_fine_elements_per_subdomain_and_dim,
                          num_subdomains[1]*half_num_fine_elements_per_subdomain_and_dim],
            num_refinements=2,
            num_partitions=num_subdomains,
            num_oversampling_layers=2*half_num_fine_elements_per_subdomain_and_dim,
            inner_boundary_segment_index=inner_boundary_segment_index)


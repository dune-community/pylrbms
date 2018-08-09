#!/usr/bin/env python

import sys
from mpi4py import MPI
import numpy

def test_make_grid():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    comm_split = comm.Split(color=(rank % 2), key=rank )
    split_rank = comm_split.Get_rank()
    split_size = comm_split.Get_size()

    print('old rank: {rank} \told size: {size}'.format(**locals()))
    print('new rank: {split_rank} \tnew size: {split_size}'.format(**locals()))

    from dune.xt.grid.provider import make_cube_grid__2d_cube_yaspgrid as make_grid

    domain=([0, 0], [1, 1])
    grid = make_grid(domain[0], domain[1], num_elements=[32, 32], num_refinements=0, overlap_size=[1, 1],
                     mpi_comm=comm_split)

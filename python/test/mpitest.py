#!/usr/bin/env python

import sys
import tempfile

from mpi4py import MPI
import numpy

from dune.xt.common.test import runmodule

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


def test_new_comm():
    if MPI.COMM_WORLD.Get_size() == 1:
        mpi_comm = MPI.COMM_WORLD.Clone()
    else:
        comm = MPI.COMM_WORLD.Clone()
        mpi_comm = comm.Split(color=(comm.Get_rank() == 0), key=comm.Get_rank())
        assert mpi_comm.Get_size() < MPI.COMM_WORLD.Get_size()
    mpi_comm.Set_errhandler(MPI.ERRORS_ARE_FATAL)


def test_blockspace():
    from pymor.vectorarrays.block import BlockVectorSpace, BlockVectorArray
    from pymor.core.pickle import dump, load

    b = BlockVectorSpace([])
    with tempfile.TemporaryFile('wb') as dp_file:
        dump(b, file=dp_file)

if __name__ == '__main__':
    runmodule(__file__)
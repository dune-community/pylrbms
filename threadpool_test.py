#!/usr/bin/env python

import sys
import time
import numpy as np

from dune.xt.common import init_mpi
init_mpi()

from dune.xt.la import (
        IstlRowMajorSparseMatrixDouble as Matrix,
        IstlDenseVectorDouble as Vector,
        SparsityPatternDefault as Pattern
        )

from concurrent.futures import ThreadPoolExecutor

N = int(sys.argv[1])
S = int(sys.argv[2])
M = int(sys.argv[3])
W = int(sys.argv[4])

print('computing {}x{} unit matrix with {} entries per row ... '.format(N, N, S), end='')
sys.stdout.flush()
t = time.time()

pattern = Pattern(N)
for ii in range(N):
    pattern.insert(ii, ii)
for jj in range(1, S): # stencil:
    for ii in range(jj, N):
        pattern.insert(ii, ii - jj)
    for ii in range(N - jj):
        pattern.insert(ii, ii + jj)
pattern.sort()

mat = Matrix(N, N, pattern)
for ii in range(N):
    mat.unit_row(ii)

print('done (took {}s)'.format(time.time() - t))
print('preparing {} input vectors ... '.format(M), end='')
sys.stdout.flush()

t = time.time()
Us = [Vector(N, ii) for ii in range(M)]
#Vs = [Vector(N, 0.) for ii in range(M)]

print('done (took {}s)'.format(time.time() - t))

print('doing mv with {} threads ... '.format(W), end='')
sys.stdout.flush()

def do_work(ii):
    V = Us[ii].copy()
    mat.mv(Us[ii], V)
    return V

t = time.time()
with ThreadPoolExecutor(max_workers=W) as executor:
    Vs = [V for V in executor.map(do_work, range(M))]

print('done (took {}s)'.format(time.time() - t))

assert np.allclose([U.sup_norm() for U in Us], [V.sup_norm() for V in Vs])


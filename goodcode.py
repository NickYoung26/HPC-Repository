#!/opt/software/anaconda/python-3.10.9/bin/python

"""
Version: Python 3.10.9

Description: Optimising parralesisation of an integral calculation equal to pi.

Date: 02/02/2026

Author: Nicholas Young
"""

from mpi4py import MPI  # pylint: disable=import-error

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

N = 100000000
DELTA = 1.0 / N

I_LOCAL = 0.0

for ii in range(rank, N, nproc):
    x = (ii + 0.5) * DELTA
    I_LOCAL += 4.0 / (1.0 + x*x)
I_LOCAL *= DELTA

I_FINAL = comm.reduce(I_LOCAL, op=MPI.SUM, root=0)

if rank == 0:
    print(f"Good Code Integral {I_FINAL:.10f}")

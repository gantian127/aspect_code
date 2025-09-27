#!/bin/bash
set -e

# A first parallel landlab example
cd mpi_landlab
mpirun -n 4 python mpi_landlab.py
cd ..

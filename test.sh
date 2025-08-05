#!/bin/bash
set -e

# A first parallel landlab example
cd mpi_landlab
mpirun -n 2 python mpi_landlab3.py
cd ..

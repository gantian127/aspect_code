#!/bin/bash
set -e

# A first parallel landlab example
cd mpi-landlab-test
mpirun -n 2 python mpi_landlab3.py
cd ..

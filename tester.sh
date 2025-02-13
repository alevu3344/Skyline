#!/bin/bash


make clean

echo "Compiling the MPI and OpenMP versions..."
make

echo "Running the tests for the OpenMP versions..."


echo "OpenMP strong scaling test"
./omp_strong_scaling.sh


echo "OpenMP weak scaling test"
./omp_weak_scaling.sh

echo "Running the tests for the MPI versions..."

echo "MPI strong scaling test"
./mpi_strong_scaling.sh

echo "MPI weak scaling test"
./mpi_weak_scaling.sh

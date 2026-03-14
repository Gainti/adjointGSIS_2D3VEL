#! /bin/bash
mpirun -np 4 ./build/solver --case cases/demo > log 2>&1 
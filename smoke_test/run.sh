#!/bin/bash
mpirun -n 1 python -u -O -m mpi4py flame1d.py -i run_params.yaml

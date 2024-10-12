#!/bin/bash
path=$1_$PMI_RANK
args="${@:2}"
nsys profile -t openmp,nvtx,ucx,mpi,osrt --mpi-impl openmpi --stats=true -f true -o $path $args

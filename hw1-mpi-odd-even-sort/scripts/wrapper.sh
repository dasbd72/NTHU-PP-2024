#!/bin/bash
path=$1_$(printf "%02d" $PMI_RANK)
args="${@:2}"
nsys profile -t openmp,nvtx,mpi --mpi-impl openmpi --stats=true -f true -o $path $args

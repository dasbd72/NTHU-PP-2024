#!/bin/sh
# Copy content of lab2.cc to lab2_hybrid.cc, lab2_omp.cc, lab2_pthread.cc
cat lab2.cc >| lab2_hybrid.cc
cat lab2.cc >| lab2_omp.cc
cat lab2.cc >| lab2_pthread.cc

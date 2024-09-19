#!/bin/sh
export TIMING=0
export DEBUG=0
export SANITIZE=0

if [ $SANITIZE -eq 1 ]; then
    export LSAN_OPTIONS=verbosity=1:log_threads=1
elif [ $LSAN_OPTIONS ]; then
    unset LSAN_OPTIONS
fi

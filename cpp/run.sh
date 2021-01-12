#!/bin/bash

make clean
make -j12

for NUM_THREADS in 1 2 4 8; do
  for i in $(seq 2); do
    echo -n "num_threads=$NUM_THREADS | "
    OMP_NUM_THREADS=$NUM_THREADS ./main
  done
done
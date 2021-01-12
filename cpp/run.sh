#!/bin/bash

make clean
make -j12

for NUM_THREADS in 2 4 8; do
  for i in $(seq 2); do
    echo -n "num_threads=$NUM_THREADS | "
    OMP_NUM_THREADS=$NUM_THREADS ./main
  done
done

# original:
# num_threads=1 | final_cost = 159997104.000000 | elapsed = 3285.131104
# num_threads=1 | final_cost = 159997104.000000 | elapsed = 3284.781006
# num_threads=2 | final_cost = 159997104.000000 | elapsed = 2144.450928
# num_threads=2 | final_cost = 159997104.000000 | elapsed = 2134.512939
# num_threads=4 | final_cost = 159997104.000000 | elapsed = 1508.692993
# num_threads=4 | final_cost = 159997104.000000 | elapsed = 1513.552979
# num_threads=8 | final_cost = 159997104.000000 | elapsed = 1820.090942
# num_threads=8 | final_cost = 159997104.000000 | elapsed = 1788.469971



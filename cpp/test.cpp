
#pragma GCC diagnostic ignored "-Wunused-result"

#include <stdio.h>
#include <float.h>
#include <iostream>
#include <random>
#include <chrono>
#include "omp.h"

#define SEED 123123
std::default_random_engine generator(SEED);

typedef int Int;
typedef float Real;

const Int n_bidders = 10000;
const Int n_items   = 10000;
const Int max_cost  = 100000;

struct T {
    Int idx1;
    Real val1;
    
    Int idx2;
    Real val2;
};

struct T T_min(struct T a, struct T b) {
    if(a.val1 < b.val1) {
      if(a.val2 <= b.val1) {
        return a;
      } else {
        return {a.idx1, a.val1, b.idx1, b.val1};
      }
      
    } else {
      if(b.val2 <= a.val1) {
        return b;
      } else {
        return {b.idx1, b.val1, a.idx1, a.val1};
      }
    }
};

#pragma omp declare reduction(T_min: struct T: omp_out=T_min(omp_in, omp_out))\
    initializer(omp_priv={0, FLT_MAX, 0, FLT_MAX})

using namespace std::chrono;

void uniform_random_problem(Real* cost_matrix) {
  std::uniform_int_distribution<Int> distribution(0, max_cost);

  for (long i = 0; i < n_items * n_bidders; i++) {
    cost_matrix[i] = (Real)distribution(generator);
  }
}


int main(int argc, char *argv[]) {
  Real* cost_matrix = (Real*)malloc(n_items * n_bidders * sizeof(Real));
  uniform_random_problem(cost_matrix);
  std::cout << "uniform_random_problem: done" << std::endl;
  
  auto t_start = high_resolution_clock::now();
  
  struct T* results = (T*)malloc(n_bidders * sizeof(T));

  // --
  // Parallelize over bidders
  // #pragma omp parallel for
  // for(Int bidder = 0; bidder < 10; bidder++) {
  //   Real min_val = (Real)max_cost + 1;
  //   for(Int i = 0; i < n_items; i++) {
  //     if(cost_matrix[n_bidders * bidder + i] < min_val) {
  //       min_val = cost_matrix[n_bidders * bidder + i];
  //     }
  //   }
  //   results[bidder] = min_val;
  // }
  
  // --
  // Parallelize over items
  
  std::uniform_int_distribution<Int> bidder_sampler(0, n_bidders);
  
  for(Int iter = 0; iter < 1000; iter++) { // Cannot be parallelized
      Int bidder = bidder_sampler(generator);
      
      struct T min_val  = {0, FLT_MAX, 0, FLT_MAX};
      
      #pragma omp parallel for reduction(T_min:min_val) default(none) shared(cost_matrix, bidder)
      for(Int i = 0; i < n_items; i++) {
        Real val = cost_matrix[n_bidders * bidder + i];
        if(val < min_val.val1) {
          min_val.val2 = min_val.val1;
          min_val.val1 = val;
        } else if(val < min_val.val2) {
          min_val.val2 = val;
        }
      }
      
      results[bidder] = min_val;
  }
  
  auto t_end   = high_resolution_clock::now();
  auto elapsed = duration_cast<microseconds>(t_end - t_start).count();
  
  Real acc1 = 0;
  Real acc2 = 0;
  for(Int i = 0; i < n_bidders; i++) {
    acc1 += results[i].val1;
    acc2 += results[i].val2;
  }
  
  std::cout << "elapsed: " << elapsed << " | acc1: " << acc1 << " | acc2: " << acc2 << std::endl;
}
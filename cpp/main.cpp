#pragma GCC diagnostic ignored "-Wunused-result"

#include <stdio.h>
#include <random>

#include "auction.hpp"

#define SEED 123123

std::default_random_engine generator(SEED);

typedef int Int;
typedef double Real;

const Int n_bidders = 16000;
const Int n_items   = 16000;
const Int max_cost  = 10000;
const Real eps      = 0.1;

// --
// Helpers

void uniform_random_problem(Real* cost_matrix) {
  std::uniform_real_distribution<Real> distribution(0.0, (Real)max_cost);

  for (long i = 0; i < n_items * n_bidders; i++) {
    cost_matrix[i] = distribution(generator);
  }
}

void save_matrix(Real* cost_matrix) {  
  // printf("save_matrix: start\n");
  
  FILE* f;
  f = fopen("prob.bin", "wb");
  fwrite(cost_matrix, sizeof(Real), n_bidders * n_items, f);
  fclose(f);
  
  // printf("save_matrix: done\n");
}

void load_matrix(Real* cost_matrix) {  
  // printf("load_matrix: start\n");
  
  FILE* f;
  f = fopen("prob.bin", "rb");
  fread(cost_matrix, sizeof(Real), n_bidders * n_items, f);
  fclose(f);
  
  // printf("load_matrix: done\n");
}

// --
// Main

int main(int argc, char *argv[]) {
  
  // --
  // Generate problem  
  
  Real* cost_matrix = (Real*)malloc(n_items * n_bidders * sizeof(Real));
  uniform_random_problem(cost_matrix);
  save_matrix(cost_matrix);
  // load_matrix(cost_matrix);
  
  // --
  // Init
  
  Int* bidder2item = (Int*)malloc(n_bidders * sizeof(Int));
  
  // --
  // Solve problem (block)
  
  for(Int i = 0; i < n_bidders; i++) bidder2item[i] = -1;
  
  auto block_start   = high_resolution_clock::now();
  auto block_cost    = auction<Int, Real>(cost_matrix, n_bidders, n_items, eps, bidder2item, 1);
  auto block_stop    = high_resolution_clock::now();
  auto block_elapsed = duration_cast<microseconds>(block_stop - block_start).count();
  
  // --
  // Solve problem (single)
  
  for(Int i = 0; i < n_bidders; i++) bidder2item[i] = -1; // reset
  
  auto single_start   = high_resolution_clock::now();
  auto single_cost    = auction<Int, Real>(cost_matrix, n_bidders, n_items, eps, bidder2item, 2);
  auto single_stop    = high_resolution_clock::now();
  auto single_elapsed = duration_cast<microseconds>(single_stop - single_start).count();
  
  // --
  // Eval
  
  // printf("block_cost = %f | single_cost = %f | block_elapsed = %f | single_elapsed = %f\n", 
  //   block_cost, single_cost, float(block_elapsed) / 1000, float(single_elapsed) / 1000);
}
#pragma GCC diagnostic ignored "-Wunused-result"

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <random>
#include "auction.hpp"

#define SEED 123123

std::default_random_engine generator(SEED);

typedef int Int;
typedef double Real;

const Int n_bidders = 40000;
const Int n_items   = 40000;
const Int max_cost  = 10000;
const Real eps      = 0.5;

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
  printf(" . | ");
}

// --
// Main

int main(int argc, char *argv[]) {
  
  Int* bidder2item  = (Int*)malloc(n_bidders * sizeof(Int));
  Real* cost_matrix = (Real*)malloc(n_items * n_bidders * sizeof(Real));

  // --
  // Generate problem  
  // uniform_random_problem(cost_matrix);
  // save_matrix(cost_matrix);
  load_matrix(cost_matrix);
  
  // --
  // Solve problem
  
  long long elapsed = auction<Int, Real>(cost_matrix, n_bidders, n_items, eps, bidder2item);
  
  // --
  // Eval
  
  Real final_cost = 0.0;
  for(Int bidder = 0; bidder < n_bidders; bidder++) {
    final_cost += cost_matrix[n_bidders * bidder + bidder2item[bidder]];
  }
  printf("final_cost=%f | elapsed=%f\n", final_cost, (float)elapsed / 1e3);
}

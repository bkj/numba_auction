
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
const Int max_cost  = 10;

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
  
  Real acc = 0;
  for(Int i = 0; i < n_bidders * n_items; i++) {
    acc += cost_matrix[i];
  }
  
  auto t_end   = high_resolution_clock::now();
  auto elapsed = duration_cast<microseconds>(t_end - t_start).count();
  std::cout << "elapsed: " << elapsed << " | acc: " << acc << std::endl;
}
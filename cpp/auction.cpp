#include <stdio.h>
#include <random>
#include "omp.h"
#include <chrono>

#define SEED 123123

using namespace std::chrono;

std::default_random_engine generator(SEED);

typedef int Int;
typedef float Real;

Int n_bidders = 1000;
Int n_items   = 1000;
Int max_cost  = 1000;
Real eps      = 0.5;
Real* cost_matrix;
Int* bidder2item;

void uniform_random_problem() {
  std::uniform_int_distribution<Int> distribution(0, max_cost);

  for (long i = 0; i < n_items * n_bidders; i++) {
    cost_matrix[i] = (Real)distribution(generator);
  }
}

void save_matrix() {  
  // for(int i = 0; i < n_bidders; i++) {
  //   for(int j = 0; j < n_bidders; j++) {
  //     printf("%f ", cost_matrix[n_bidders * i + j]);
  //   }
  //   printf("\n");
  // }
  
  printf("save_matrix: start\n");
  FILE* f;
  f = fopen("prob.bin", "wb");
  fwrite(cost_matrix, sizeof(Real), n_bidders * n_items, f);
  fclose(f);
  printf("save_matrix: done\n");
}

template<typename T>
void fill(T* x, Int n, T val) {
  for(Int i = 0; i < n; i++) x[i] = val;
}

void auction() {
  Real* cost       = (Real*)malloc(n_items * sizeof(Real));
  Real* high_bids  = (Real*)malloc(n_items * sizeof(Real));
  Int* high_bidder = (Int*)malloc(n_items * sizeof(Int));
  Int* item2bidder = (Int*)malloc(n_items * sizeof(Int));
  
  fill<Real>(cost, n_items, 0);
  fill<Real>(high_bids, n_items, -1);
  fill<Int>(high_bidder, n_items, -1);
  fill<Int>(item2bidder, n_items, -1);

  Int* idx1        = (Int*)malloc(n_bidders * sizeof(Int));
  Int* idx2        = (Int*)malloc(n_bidders * sizeof(Int));
  Real* val1       = (Real*)malloc(n_bidders * sizeof(Real));
  Real* val2       = (Real*)malloc(n_bidders * sizeof(Real));
  
  fill<Int>(idx1, n_bidders, -1);
  fill<Int>(idx2, n_bidders, -1);
  fill<Real>(val1, n_bidders, -1);
  fill<Real>(val2, n_bidders, -1);
  fill<Int>(bidder2item, n_bidders, -1);
  
  Int unassigned_bidders = n_bidders;
  while(unassigned_bidders > 0) {
    
    fill<Int>(idx1, n_bidders, -1);
    fill<Int>(idx2, n_bidders, -1);
    fill<Real>(val1, n_bidders, -1);
    fill<Real>(val2, n_bidders, -1);
  
    // --
    // Bid
    
    for(Int bidder = 0; bidder < n_bidders; bidder++) {
      if(bidder2item[bidder] != -1) continue;
      for(Int item = 0; item < n_items; item++) {
        Real val = cost_matrix[n_bidders * bidder + item] - cost[item];
        if(val > val1[bidder]) {
          idx2[bidder] = idx1[bidder];
          idx1[bidder] = item;
          val2[bidder] = val1[bidder];
          val1[bidder] = val;
        } else if(val > val2[bidder]) {
          idx2[bidder] = item;
          val2[bidder] = val;
        }
      }
    }
    
    // --
    // Compete

    fill<Real>(high_bids, n_items, -1);
    fill<Int>(high_bidder, n_items, -1);

    for(Int bidder = 0; bidder < n_bidders; bidder++) {
      if(bidder2item[bidder] != -1) continue;
      
      Real bid = val1[bidder] - val2[bidder] + eps;
      if(bid > high_bids[idx1[bidder]]) {
        high_bids[idx1[bidder]]   = bid;
        high_bidder[idx1[bidder]] = bidder;
      }
    }
    
    // --
    // Assign
    
    for(Int item = 0; item < n_items; item++) {
      if(high_bids[item] == -1) continue;
      
      cost[item] += high_bids[item];
      
      if(item2bidder[item] != -1) {
        bidder2item[item2bidder[item]] = -1;
        unassigned_bidders++;
      }
      
      item2bidder[item] = high_bidder[item];
      bidder2item[high_bidder[item]] = item;
      unassigned_bidders--;
    }
  }
}

int main(int argc, char *argv[]) {
  
  // --
  // Generate problem
  
  cost_matrix = (Real*)malloc(n_items * n_bidders * sizeof(Real));
  uniform_random_problem();
  save_matrix();
  
  // --
  // Solve problem
  
  printf("---\n");
  
  bidder2item = (Int*)malloc(n_bidders * sizeof(Int));
  
  auto t1 = high_resolution_clock::now();
  auction();
  long long elapsed = duration_cast<microseconds>(high_resolution_clock::now() - t1).count();
  
  // --
  // Eval
  
  Real final_cost = 0;
  for(Int bidder = 0; bidder < n_bidders; bidder++) {
    final_cost += cost_matrix[n_bidders * bidder + bidder2item[bidder]];
  }
  printf("final_cost = %f | elapsed = %f\n", final_cost, float(elapsed) / 1000);
}
#pragma GCC diagnostic ignored "-Wunused-result"

// #include <pybind11/pybind11.h>
// #include <torch/extension.h>
#include "tmp.h"

#define SEED 123123

std::default_random_engine generator(SEED);

const Int n_bidders = 32768;
const Int n_items   = 32768;
const Int max_cost  = 100000000;
const Real eps      = 0.5;

// --
// Helpers

void uniform_random_problem(Real* cost_matrix) {
  std::uniform_int_distribution<Int> distribution(0, max_cost);

  for (long i = 0; i < n_items * n_bidders; i++) {
    cost_matrix[i] = (Real)distribution(generator);
  }
}

void save_matrix(Real* cost_matrix) {  
  printf("save_matrix: start\n");
  
  FILE* f;
  f = fopen("prob.bin", "wb");
  fwrite(cost_matrix, sizeof(Real), n_bidders * n_items, f);
  fclose(f);
  
  printf("save_matrix: done\n");
}

void load_matrix(Real* cost_matrix) {  
  printf("load_matrix: start\n");
  
  FILE* f;
  f = fopen("prob.bin", "rb");
  fread(cost_matrix, sizeof(Real), n_bidders * n_items, f);
  fclose(f);
  
  printf("load_matrix: done\n");
}

// --
// Auction algorithm

long long auction(Real* cost_matrix, Int* bidder2item) {
  auto t_start = high_resolution_clock::now();
  
  Real* cost       = (Real*)malloc(n_items * sizeof(Real));
  Real* high_bids  = (Real*)malloc(n_items * sizeof(Real));
  Int* high_bidder = (Int*)malloc(n_items * sizeof(Int));
  Int* item2bidder = (Int*)malloc(n_items * sizeof(Int));
  
  fill<Real>(cost, n_items, 0);
  fill<Real>(high_bids, n_items, -1);
  fill<Int>(high_bidder, n_items, -1);
  fill<Int>(item2bidder, n_items, -1);
  fill<Int>(bidder2item, n_bidders, -1);
  
  Int* idx1  = (Int*)malloc(n_bidders * sizeof(Int));
  Int* idx2  = (Int*)malloc(n_bidders * sizeof(Int));
  Real* val1 = (Real*)malloc(n_bidders * sizeof(Real));
  Real* val2 = (Real*)malloc(n_bidders * sizeof(Real));
  
  Int unassigned_bidders = n_bidders;
  Int loop_counter = 0;
  
  fill<Int>(idx1, n_bidders, -1);
  fill<Int>(idx2, n_bidders, -1);
  fill<Real>(val1, n_bidders, -1);
  fill<Real>(val2, n_bidders, -1);

  while(unassigned_bidders > 64) {
    printf("%d ", unassigned_bidders);
    auto t1 = high_resolution_clock::now();
  
    // --
    // Bid
    
    // This definitely helps in early iterations. May add overhead in later iterations
    // Might also be a good idea to use a different datastructure, to avoid iterating over _all_ bidders in later stages
    #pragma omp parallel for
    for(Int bidder = 0; bidder < n_bidders; bidder++) {
      if(bidder2item[bidder] != -1) continue;

      Int idx2_  = -1;
      Int idx1_  = -1;
      Real val2_ = -1;
      Real val1_ = -1;
      
      for(Int item = 0; item < n_items; item++) {
        Real val = cost_matrix[n_bidders * bidder + item] - cost[item];
        if(val > val1_) {
          idx2_ = idx1_;
          idx1_ = item;
          val2_ = val1_;
          val1_ = val;
        } else if(val > val2_) {
          idx2_ = item;
          val2_ = val;
        }
      }
      
      idx2[bidder] = idx2_;
      idx1[bidder] = idx1_;
      val2[bidder] = val2_;
      val1[bidder] = val1_;
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
    long long loop_time = duration_cast<microseconds>(high_resolution_clock::now() - t1).count();
    printf("%d %lld \n", loop_counter, loop_time);
    loop_counter++;
  }
  
  return duration_cast<microseconds>(high_resolution_clock::now() - t_start).count();
}

int main(int argc, char *argv[]) {
  
  Int* bidder2item  = (Int*)malloc(n_bidders * sizeof(Int));
  Real* cost_matrix = (Real*)malloc(n_items * n_bidders * sizeof(Real));

  // --
  // Generate problem  
  uniform_random_problem(cost_matrix);
  save_matrix(cost_matrix);
  // load_matrix();
  
  // --
  // Solve problem
  
  long long elapsed = auction(cost_matrix, bidder2item);
  
  // --
  // Eval
  
  Real final_cost = 0;
  for(Int bidder = 0; bidder < n_bidders; bidder++) {
    final_cost += cost_matrix[n_bidders * bidder + bidder2item[bidder]];
  }
  printf("final_cost = %f | elapsed = %f\n", final_cost, float(elapsed) / 1000);
}

// #define _PYTHON_INTERFACE
// #ifdef _PYTHON_INTERFACE

// long long py_auction(
//   torch::Tensor cost_matrix,
//   torch::Tensor bidder2item
// ) {
//   return auction(
//     cost_matrix.data_ptr<Real>(),
//     bidder2item.data_ptr<Int>()
//   );
// }

// PYBIND11_MODULE(auction, m) {
//   m.def("auction", py_auction);
// }
// #endif
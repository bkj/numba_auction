#ifndef __AUCTION_H__
#define __AUCTION_H__

#pragma GCC diagnostic ignored "-Wunused-result"

#include "iostream"
#include <malloc.h>
#include <stdio.h>
#include <chrono>
#include "omp.h"

using namespace std::chrono;

#define VERBOSE

template<typename Int, typename Real>
struct T {
    Int idx1;
    Real val1;
    
    Int idx2;
    Real val2;
};


template<typename T>
T T_max(T a, T b) {
    if(a.val1 >= b.val1) {
      if(a.val2 >= b.val1) {
        return a;
      } else {
        return {a.idx1, a.val1, b.idx1, b.val1};
      }
      
    } else {
      if(b.val2 > a.val1) {
        return b;
      } else {
        return {b.idx1, b.val1, a.idx1, a.val1};
      }
    }
};

#pragma omp declare reduction(T_max: struct T<int, double>: omp_out=T_max(omp_in, omp_out))\
    initializer(omp_priv={-1, -1, -1, -1})

// <<

template<typename Val, typename Int>
void fill(Val* x, Int n, Val val) {
  for(Int i = 0; i < n; i++) x[i] = val;
}


template <typename Int, typename Real>
long long auction(Real* cost_matrix, Int n_bidders, Int n_items, Real eps, Int* bidder2item) {
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

#ifdef VERBOSE
  Int loop_counter = 0;
#endif
  
  fill<Int>(idx1, n_bidders, -1);
  fill<Int>(idx2, n_bidders, -1);
  fill<Real>(val1, n_bidders, -1);
  fill<Real>(val2, n_bidders, -1);

  while(unassigned_bidders > 0) {

#ifdef VERBOSE
    printf("%d ", unassigned_bidders);
    auto t1 = high_resolution_clock::now();
#endif
  
    // --
    // Bid
        
    // Int thresh = -1;
    Int thresh = 8;
    
    if(unassigned_bidders > thresh) {
      #pragma omp parallel for
      for(Int bidder = 0; bidder < n_bidders; bidder++) { // This may give non-uniform work to threads
        if(bidder2item[bidder] != -1) continue;

        Int idx2_  = -1;
        Int idx1_  = -1;
        Real val2_ = -1;
        Real val1_ = -1;
        
        for(Int item = 0; item < n_items; item++) {
          Real val = cost_matrix[n_bidders * bidder + item] - cost[item];
          if(val > val1_) {
            idx2_ = idx1_;
            val2_ = val1_;
            
            idx1_ = item;
            val1_ = val;
          } else if(val > val2_) {
            idx2_ = item;
            val2_ = val;
          }
        }
        
        idx1[bidder] = idx1_;
        val1[bidder] = val1_;
        idx2[bidder] = idx2_;
        val2[bidder] = val2_; 
      }
      
    } else {
      
      // For sufficiently large problems, this seems to help
      
      for(Int bidder = 0; bidder < n_bidders; bidder++) {
        if(bidder2item[bidder] != -1) continue;

        struct T<Int, Real> acc  = {-1, -1, -1, -1};
        
        #pragma omp parallel for reduction(T_max:acc) default(none) shared(n_bidders, n_items, cost_matrix, bidder, cost) num_threads(4)
        for(Int item = 0; item < n_items; item++) {
          Real val = cost_matrix[n_bidders * bidder + item] - cost[item];
          if(val > acc.val1) {
            acc.idx2 = acc.idx1;
            acc.val2 = acc.val1;
            
            acc.idx1 = item;
            acc.val1 = val;
          } else if(val > acc.val2) { // Tiebreaking is needed, I think
            acc.idx2 = item;
            acc.val2 = val;
          }
        }

        idx1[bidder] = acc.idx1;
        val1[bidder] = acc.val1;
        idx2[bidder] = acc.idx2;
        val2[bidder] = acc.val2;
      }
    }
    
    // --
    // Compete

    // fill<Real>(high_bids, n_items, -1);
    // fill<Int>(high_bidder, n_items, -1);

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
      
      high_bids[item]   = -1;
      high_bidder[item] = -1;
    }
#ifdef VERBOSE
    long long loop_time = duration_cast<microseconds>(high_resolution_clock::now() - t1).count();
    printf("%d %lld \n", loop_counter, loop_time);
    loop_counter++;
#endif
  }
  
  return duration_cast<microseconds>(high_resolution_clock::now() - t_start).count();
}

#endif
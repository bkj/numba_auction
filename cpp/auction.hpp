#ifndef __AUCTION_H__
#define __AUCTION_H__

#pragma GCC diagnostic ignored "-Wunused-result"

#include <malloc.h>
#include <stdio.h>
#include <chrono>
#include "omp.h"

using namespace std::chrono;

// --
// Helpers

template<typename Int, typename Real>
struct T {
    Int idx1;
    Real val1;
    Real val2;
};


template<typename T>
T T_max(T a, T b) {
    if(a.val1 >= b.val1) {
      if(a.val2 >= b.val1) {
        return a;
      } else {
        return {a.idx1, a.val1, b.val1};
      }
      
    } else {
      if(b.val2 > a.val1) {
        return b;
      } else {
        return {b.idx1, b.val1, a.val1};
      }
    }
};

#pragma omp declare reduction(T_max: struct T<int, double>: omp_out=T_max(omp_in, omp_out))\
    initializer(omp_priv={-1, -1, -1})

template<typename Val, typename Int>
void fill(Val* x, Int n, Val val) {
  for(Int i = 0; i < n; i++) x[i] = val;
}

template <typename T, typename Int>
T* malloc_and_fill(Int n, T val) {
  T* x = (T*)malloc(n * sizeof(T));
  for(Int i = 0; i < n; i++) x[i] = val;
  return x;
}

// ------------------------
// Auction Algorithm (hybrid)

template <typename Int, typename Real>
Real block_auction(
  Real* cost_matrix, 
  Int n_bidders, 
  Int n_items, 
  Real eps, 
  Int* bidder2item,
  Int* item2bidder,
  Real* cost,
  Int thresh=0
) {
  
  Real* high_bids   = malloc_and_fill<Real>(n_items, -1);
  Int*  high_bidder = malloc_and_fill<Int>(n_items, -1);
  Int*  idx1        = malloc_and_fill<Int>(n_bidders, -1);
  Real* val1        = malloc_and_fill<Real>(n_bidders, -1);
  Real* val2        = malloc_and_fill<Real>(n_bidders, -1);

  Real score = 0;

  auto t0 = high_resolution_clock::now();
  
  Int unassigned_bidders = n_bidders;
  while(unassigned_bidders > thresh) {
    
    // --
    // Bid
    
    #pragma omp parallel for
    for(Int bidder = 0; bidder < n_bidders; bidder++) {
      if(bidder2item[bidder] != -1) continue;

      Int idx1_  = -1;
      Real val1_ = -1;
      Real val2_ = -1;
      
      for(Int item = 0; item < n_items; item++) {
        Real val = cost_matrix[bidder * n_items + item] - cost[item];
        if(val > val1_) {
          val2_ = val1_;
          val1_ = val;
          idx1_ = item;
        } else if(val > val2_) {
          val2_ = val;
        }
      }
      
      idx1[bidder] = idx1_;
      val2[bidder] = val2_;
      val1[bidder] = val1_;
    }
    
    // --
    // Compete

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
        score -= cost_matrix[item2bidder[item] * n_items + item];
        bidder2item[item2bidder[item]] = -1;
        unassigned_bidders++;
      }
      
      item2bidder[item] = high_bidder[item];
      bidder2item[high_bidder[item]] = item;
      score += cost_matrix[high_bidder[item] * n_items + item];
      
      unassigned_bidders--;
      
      high_bids[item]   = -1;
      high_bidder[item] = -1;
    }

    auto t1 = high_resolution_clock::now();
    auto ee = duration_cast<microseconds>(t1 - t0).count();
    
    printf("block %d %f %ld\n", thresh, score, ee);
  }
  
  return score;
}

template <typename Int, typename Real>
Real single_auction(
  Real* cost_matrix, 
  Int n_bidders, 
  Int n_items, 
  Real eps, 
  Int* bidder2item,
  Int* item2bidder,
  Real* cost,
  Int thresh=0
) {
  
  Real score = 0;
  
  auto t0 = high_resolution_clock::now();
  
  for(Int start = 0; start < n_bidders; start++) { // This may give non-uniform work to threads
    if(bidder2item[start] != -1) continue;
    
    Int bidder = start;
    
    while(true) {
      
      struct T<Int, Real> acc  = {-1, -1, -1};
      #pragma omp parallel for reduction(T_max:acc) default(none) shared(n_bidders, n_items, cost_matrix, bidder, cost)
      for(Int item = 0; item < n_items; item++) {
        Real val = cost_matrix[bidder * n_items + item] - cost[item];
        if(val > acc.val1) {
          acc.val2 = acc.val1;
          acc.val1 = val;
          acc.idx1 = item;
        } else if(val > acc.val2) { // Tiebreaking is sometimes needed
          acc.val2 = val;
        }
      }
      
      cost[acc.idx1] += acc.val1 - acc.val2 + eps;;
      
      bool had_bidder = item2bidder[acc.idx1] != -1;
      
      Int prev_bidder;
      if(had_bidder) {
        prev_bidder = item2bidder[acc.idx1];
        bidder2item[prev_bidder] = -1;
        score -= cost_matrix[prev_bidder * n_items + acc.idx1];
      }
      
      bidder2item[bidder]   = acc.idx1;
      item2bidder[acc.idx1] = bidder;
      score += cost_matrix[bidder * n_items + acc.idx1];
      
      if(!had_bidder) break;
      bidder = prev_bidder;
    }

    auto t1 = high_resolution_clock::now();
    auto ee = duration_cast<microseconds>(t1 - t0).count();

    printf("single -1 %f %ld\n", score, ee);
  }
  
  return score;
}


template <typename Int, typename Real>
Real auction(Real* cost_matrix, Int n_bidders, Int n_items, Real eps, Int* bidder2item, Int flag) {
  
  Real* cost       = malloc_and_fill<Real>(n_items, 0);
  Int* item2bidder = malloc_and_fill<Int>(n_items, -1);
  
  if(flag == 1) {
    return block_auction(
      cost_matrix, 
      n_bidders, 
      n_items, 
      eps, 
      bidder2item,
      item2bidder,
      cost
    );
  } else if(flag == 2) {
    return single_auction(
      cost_matrix, 
      n_bidders, 
      n_items, 
      eps, 
      bidder2item,
      item2bidder,
      cost
    );
  } else {
    printf("ERROR: unknown flag %d \n", flag);
    return -1;
  }
}

#endif
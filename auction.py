#!/usr/bin/env python

"""
    auction_vs_jv.py
"""

import sys
import json
import numpy as np
from time import perf_counter_ns
from lap import lapjv
from numba import njit, prange

@njit()
def auction(X, eps=1):
    n_bidders, n_items = X.shape
    
    cost = np.zeros(n_items)
    
    idx1 = np.zeros(n_bidders, dtype=np.int32) - 1
    idx2 = np.zeros(n_bidders, dtype=np.int32) - 1
    val1 = np.zeros(n_bidders) - 1
    val2 = np.zeros(n_bidders) - 1
    
    high_bids   = np.zeros(n_items) - 1
    high_bidder = np.zeros(n_items, dtype=np.int32) - 1

    item2bidder = np.zeros(n_items,  dtype=np.int32) - 1
    bidder2item = np.zeros(n_bidders, dtype=np.int32) - 1
    
    unassigned_bidders = n_bidders
    while unassigned_bidders > 0:
        
        idx1.fill(-1)
        idx2.fill(-1)
        val1.fill(-1)
        val2.fill(-1)
        high_bids.fill(-1)
        high_bidder.fill(-1)
        
        # --
        # Bid
        
        for bidder in range(n_bidders):
            if bidder2item[bidder] != -1: continue
            
            for item in range(n_items):
                val = X[bidder, item] - cost[item]
                if val > val1[bidder]:
                    idx2[bidder] = idx1[bidder]
                    idx1[bidder] = item
                    
                    val2[bidder] = val1[bidder]
                    val1[bidder] = val
                    
                elif val > val2[bidder]:
                    idx2[bidder] = item
                    val2[bidder] = val
        
        # --
        # Compete
        
        for bidder in range(n_bidders):
            if bidder2item[bidder] != -1: continue
            
            bid = val1[bidder] - val2[bidder] + eps
            
            if bid > high_bids[idx1[bidder]]:
                high_bids[idx1[bidder]]   = bid
                high_bidder[idx1[bidder]] = bidder
        
        # --
        # Assign
        
        for item in range(n_items):
            if high_bids[item] == -1: continue
            
            cost[item] += high_bids[item]
            
            if item2bidder[item] != -1:
                bidder2item[item2bidder[item]] = -1
                unassigned_bidders += 1
            
            item2bidder[item]              = high_bidder[item]
            bidder2item[high_bidder[item]] = item
            unassigned_bidders -= 1
    
    return bidder2item, cost

# --
# Init

# n_bidders = 20_000
# n_items   = 20_000
# max_val   = 100

# X = np.random.uniform(0, max_val, (n_bidders, n_items))
# print('X: done', file=sys.stderr)

# X = np.loadtxt('cpp/prob.txt')
X = np.fromfile('cpp/prob.bin', dtype=np.float32)
n = int(np.sqrt(X.shape[0]))
X = X.reshape(n, n)

n_bidders = X.shape[0]
n_items   = X.shape[1]
max_val   = X.max()
print('X: done', file=sys.stderr)

# --
# Auction

# _ = auction(X[:100,:100])

# t = perf_counter_ns()
# bidder2item, cost = auction(X)
# auc_time = perf_counter_ns() - t

# auc_score = X[(np.arange(n_bidders), bidder2item)].sum()

# print('auc: done', file=sys.stderr)

# --
# Baseline (JV)

t = perf_counter_ns()
_, row_ind, _ = lapjv(X.max() - X)
lap_time  = perf_counter_ns() - t

lap_score = X[(np.arange(n_bidders), row_ind)].sum()

print('lap: done', file=sys.stderr)

# --
# Log

print(json.dumps({
    # "auc_score" : float(auc_score), 
    "lap_score" : float(lap_score),
    # "auc_time"  : float(auc_time / 1e6),
    "lap_time"  : float(lap_time / 1e6),
}))

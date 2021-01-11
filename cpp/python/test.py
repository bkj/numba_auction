#!/bin/bash

# test.py

import sys
import json
import torch
import numpy as np
from lap import lapjv
from time import perf_counter_ns

import sys
sys.path.append('.')
sys.path.append('build')
from auction import auction as c_auction


def auction(X, eps=0.1):
  n_bidders   = X.shape[0]
  n_items     = X.shape[1]
  bidder2item = torch.zeros(n_bidders).int()
  
  _ = c_auction(torch.FloatTensor(X), n_bidders, n_items, eps, bidder2item)
  return bidder2item.numpy(), X[(np.arange(n_bidders), bidder2item)].sum()

# --
# Make data

np.random.seed(123)

n       = 16_000
max_val = 10000
X       = np.sqrt(np.random.uniform(0, max_val, (n, n)))
print('X: done', file=sys.stderr)

# --
# Auction

t = perf_counter_ns()

auc_ass, auc_cost = auction(X)

auc_ms = (perf_counter_ns() - t) / 1e6
print(f'auc_ms={auc_ms}', file=sys.stderr)

# --
# Baseline (lapjv)

t = perf_counter_ns()

_, lap_ass, _ = lapjv(X.max() - X)
lap_cost      = X[(np.arange(X.shape[0]), lap_ass)].sum()

lap_ms = (perf_counter_ns() - t) / 1e6
print(f'lap_ms={lap_ms}', file=sys.stderr)

print(json.dumps({
  "auc_cost" : auc_cost,
  "lap_cost" : lap_cost,
  "gap"      : (lap_cost - auc_cost) / lap_cost,
  "disagree" : float((auc_ass != lap_ass).mean()),
  "auc_ms"   : auc_ms,
  "lap_ms"   : lap_ms,
}))



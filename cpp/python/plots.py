import pandas as pd
from matplotlib import pyplot as plt
from rcode import *

from glob import glob

dfs = []
for f in glob('results/*'):
  df = pd.read_csv(f, sep=' ', header=None)
  del df[3]
  df.columns = ('n_unassigned', 'loop_counter', 'loop_timer')
  df['n_threads'] = int(f.split('/')[-1])
  dfs.append(df)

df = pd.concat(dfs).reset_index(drop=True)

# --

for n_threads in sorted(df.n_threads.unique()):
  sub = df[df.n_threads == n_threads]
  sub = sub[sub.n_unassigned < 100]
  _ = plt.scatter(sub.n_unassigned, sub.loop_timer, label=n_threads, s=2, alpha=0.5)

_ = plt.legend()
_ = plt.xlabel('n_unassigned')
_ = plt.ylabel('loop_timer')
_ = plt.yscale('log')
_ = plt.xscale('log')
show_plot()

# --
# Cumulative time

for n_threads in sorted(df.n_threads.unique()):
  sub = df[df.n_threads == n_threads]
  _ = plt.plot(sub.loop_timer.cumsum().values, label=n_threads)

_ = plt.legend()
_ = plt.ylabel('loop_timer')
_ = plt.yscale('log')
show_plot()

# --

z = df.groupby(['n_unassigned', 'n_threads']).loop_timer.sum().reset_index()
z.head(20)

z[z.n_unassigned < 32].groupby('n_threads').loop_timer.sum()
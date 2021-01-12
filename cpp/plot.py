import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from rcode import *

df = pd.read_csv('res', header=None, sep=' ')
df.columns = ('method', 'thresh', 'score', 'time')

sel = (df.method == 'block') & (df.thresh > 0)
offset = df[sel].time.values[-1]
df.time[df.method == 'single'] += offset
df.method[sel] = 'single'

block  = df[df.method == 'block']
single = df[df.method == 'single']

_ = plt.plot(block.time, block.score, label='block')
_ = plt.plot(single.time, single.score, label='single')
_ = plt.legend()
_ = plt.xscale('log')
_ = plt.yscale('log')
show_plot()

best_score = block.score.values[-1]
_ = plt.plot(block.time / 1e3, (best_score - block.score) / best_score)
_ = plt.legend()
# _ = plt.xscale('log')
_ = plt.yscale('log')

show_plot()


block[block.time > 1e6].score.iloc[0]
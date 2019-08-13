import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

f = 'grnn30288-30int-1tid-1e-05a-41T-20D-4i-0.045lr-6069ms-1b-52sn.csv'
data = pd.read_csv(f'log/{f}', sep=",", header=None)
data.columns = ["datastring", "iteration", "variance_score", "mseloss", "duration"]

del data['iteration']

plt.style.use('ggplot')
plt.figure(1, figsize=(12, 7))
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.title('Loss')

# create static legend
colorA = '#41393E'
colorB = '#08A045'
patchA = mpatches.Patch(color=colorA, label='mseloss')
patchB = mpatches.Patch(color=colorB, label='variance score')
plt.legend(handles=[patchA, patchB])

iteration = 100
plt.plot(
    [i for i in range(len(data['mseloss'][:iteration]))],
    data['mseloss'][:iteration],
    color=colorA,
    linestyle='-')

# plt.plot(
#     [i for i in range(len(data['variance_score'][:iteration]))],
#     data['variance_score'][:iteration],
#     color=colorB,
#     linestyle='-')

plt.show()
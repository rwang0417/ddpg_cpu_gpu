# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import brewer2mpl
import matplotlib as mpl
from scipy import signal

nstart=0
nend=100
noisemax=100
bmap = brewer2mpl.get_map('Set2','qualitative', 7)
colors = bmap.mpl_colors
params = {
    'axes.labelsize': 22,
    'font.size': 20,
    'legend.fontsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'text.usetex': True ,
    'figure.figsize': [7, 5.5], # instead of 4.5, 4.5 now[7,5.5]
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'ps.useafm' : True,
    'pdf.use14corefonts':True,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
}
mpl.RcParams.update(params)

# y=np.array(np.loadtxt('perfcheck.txt'))
# perf=np.mean(y,axis=1)
# cstd=np.std(y,axis=1)
# step=noisemax/int(perf.shape[0]-1)
# sind=int(nstart/step)
# eind=int(nend/step)+1
# f5,=plt.plot(np.arange(sind,(eind-1)*step+1,step),perf[sind:eind],color='orange', linewidth=3)
# plt.fill_between(np.arange(sind,(eind-1)*step+1,step),(perf[sind:eind]-cstd[sind:eind]),(perf[sind:eind]+cstd[sind:eind]), alpha=0.3, color='orange')

i, x, y = np.loadtxt('perfcheck.txt', dtype=np.float64, delimiter=' ',  unpack=True, usecols=(0,1,2))
i = 100*i
plt.plot(i,x,color='orange', linewidth=3)
#plt.plot(i, (x+y), alpha=0.3, color='orange')
#plt.plot(i, (x-y), alpha=0.3, color='orange')
plt.fill_between(i, (x+y), (x-y), alpha=0.3, color='orange')

y1=np.array(np.loadtxt('clopdata.txt'))
perf1=np.mean(y1,axis=1)
cstd1=np.std(y1,axis=1)
step=noisemax/int(perf1.shape[0]-1)
sind=int(nstart/step)
eind=int(nend/step)+1
f6,=plt.plot(np.arange(sind,(eind-1)*step+1,step),perf1[sind:eind],color=colors[0], linewidth=3)
plt.fill_between(np.arange(sind,(eind-1)*step+1,step),(perf1[sind:eind]-cstd1[sind:eind]),(perf1[sind:eind]+cstd1[sind:eind]), alpha=0.3, color=colors[0])

#DDPG
i, x, y, z, a = np.loadtxt('data.txt', dtype=np.float64, delimiter=',\t',  unpack=True, usecols=(0,1,2,3,4))
i = 100*i
plt.plot(i,x,color='dodgerblue', linewidth=3)
#plt.plot(i, (x+y), alpha=0.3, color='orange')
#plt.plot(i, (x-y), alpha=0.3, color='orange')
plt.fill_between(i, (x+y), (x-y), alpha=0.3, color='dodgerblue')

print('closed-loop variance ratio={}'.format(np.mean(cstd1)/np.mean(y)))

#plt.xlabel(" Percent of max. control (Std dev of perturbed noise)", fontsize=16)
#plt.ylabel("Terminal state MSE (Avergaed over {} samples)".format(n_samples), fontsize=16)
plt.grid(axis='y', color='.910', linestyle='-', linewidth=1.5)
plt.grid(axis='x', color='.910', linestyle='-', linewidth=1.5)
#plt.tight_layout()
plt.legend(['D2C Replan','D2C','DDPG'],loc='upper left',fontsize=14)

plt.xlabel('Std dev of process noise (Percent of max. control)', fontsize=14)
plt.ylabel('L2-norm of terminal state error', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
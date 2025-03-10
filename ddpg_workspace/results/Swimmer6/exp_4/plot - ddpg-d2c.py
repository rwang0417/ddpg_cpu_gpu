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
'axes.labelsize': 20,
'font.size': 20,
'legend.fontsize': 15,
'xtick.labelsize': 15,
'ytick.labelsize': 15,
'text.usetex': True ,
'figure.figsize': [8, 6], # instead of 4.5, 4.5
'font.weight': 'bold',
'axes.labelweight': 'bold',
'ps.useafm' : True,
'pdf.use14corefonts':True,
'pdf.fonttype': 42,
'ps.fonttype': 42
 }
mpl.rcParams.update(params)

y=np.array(np.loadtxt('perfcheck0.txt'))
perf=np.mean(y,axis=1)
cstd=np.std(y,axis=1)
step=noisemax/int(perf.shape[0]-1)
sind=int(nstart/step)
eind=int(nend/step)+1
f5,=plt.plot(np.arange(sind,(eind-1)*step+1,step),perf[sind:eind],color=colors[0], linewidth=3)
plt.fill_between(np.arange(sind,(eind-1)*step+1,step),(perf[sind:eind]-cstd[sind:eind]),(perf[sind:eind]+cstd[sind:eind]), alpha=0.3, color=colors[0])

#DDPG
i, x, y, z, a = np.loadtxt('data.txt', dtype=np.float64, delimiter=',\t',  unpack=True, usecols=(0,1,2,3,4))
i = 100*i
plt.plot(i,x,color=colors[1], linewidth=3)
#plt.plot(i, (x+y), alpha=0.3, color='orange')
#plt.plot(i, (x-y), alpha=0.3, color='orange')
plt.fill_between(i, (x+y), (x-y), alpha=0.3, color=colors[1])

#plt.xlabel(" Percent of max. control (Std dev of perturbed noise)", fontsize=16)
#plt.ylabel("Terminal state MSE (Avergaed over {} samples)".format(n_samples), fontsize=16)
plt.grid(axis='y', color='.910', linestyle='-', linewidth=1.5)
plt.grid(axis='x', color='.910', linestyle='-', linewidth=1.5)
#plt.tight_layout()
plt.legend(['D2C','DDPG'],loc='upper left')
	
plt.xlabel('Std dev of perturbed noise (Percent of max. control)')
plt.ylabel('L2-norm of terminal state error')
plt.show()
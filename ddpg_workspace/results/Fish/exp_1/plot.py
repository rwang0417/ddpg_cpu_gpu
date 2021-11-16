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

y1=np.array(np.loadtxt('fish.txt'))
perf1=np.mean(y1,axis=1)
cstd1=np.std(y1,axis=1)
step=noisemax/int(perf1.shape[0]-1)
sind=int(nstart/step)
eind=int(nend/step)+1
f6,=plt.plot(np.arange(sind,(eind-1)*step+1,step),perf1[sind:eind],color='orange', linewidth=3)
plt.fill_between(np.arange(sind,(eind-1)*step+1,step),(perf1[sind:eind]-cstd1[sind:eind]),(perf1[sind:eind]+cstd1[sind:eind]),alpha=0.3,color='orange')

y=np.array(np.loadtxt('clopdata.txt'))
perf=np.mean(y,axis=1)
cstd=np.std(y,axis=1)
step=noisemax/int(perf.shape[0]-1)
sind=int(nstart/step)
eind=int(nend/step)+1
f5,=plt.plot(np.arange(sind,(eind-1)*step+1,step),perf[sind:eind],color=colors[0], linewidth=3)
plt.fill_between(np.arange(sind,(eind-1)*step+1,step),(perf[sind:eind]-cstd[sind:eind]),(perf[sind:eind]+cstd[sind:eind]),alpha=0.3,color=colors[0])
plt.grid(axis='y', color='.910', linestyle='-', linewidth=1.5)
plt.grid(axis='x', color='.910', linestyle='-', linewidth=1.5)
#plt.tight_layout()

plt.xlabel('Std dev of perturbed noise (Percent of max. control)',fontsize=14)
plt.ylabel('L2-norm of terminal state error',fontsize=14)
#plt.grid(True)

i, x, y, z, a = np.loadtxt('data1.txt', dtype=np.float64, delimiter=',\t',  unpack=True, usecols=(0,1,2,3,4))
i = 100*i
plt.plot(i,x,color='dodgerblue', linewidth=3)
plt.fill_between(i, (x+y), (x-y), alpha=0.3, color='dodgerblue')

print('closed-loop variance ratio={}'.format(np.mean(cstd[0:6])/np.mean(y)))
#     y_ddpg = np.array(np.loadtxt('data.txt')).T
#     x_ddpg = 100*np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
# #    x_ddpg = 100*np.array([0,  0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
#     print(x_ddpg, y_ddpg[1])

# ddpg = plt.plot(x_ddpg, y_ddpg[0], linewidth=3, label='DDPG', color=colors[1])
# plt.fill_between(x_ddpg, y_ddpg[0] + y_ddpg[1], y_ddpg[0] - y_ddpg[1],alpha=0.3, color=colors[1])

#plt.grid(linewidth=1.5)
# plt.legend(['D2C','DDPG'],fontsize=14)
plt.legend(['D2C Replan','D2C','DDPG'],fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, 0.1)
plt.show()
# print('averaged by {value1} rollouts'.format(value1=y.shape[1]))

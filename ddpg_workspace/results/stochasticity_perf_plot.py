# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import brewer2mpl
import matplotlib as mpl
from scipy import signal
# In[plot settings]
bmap = brewer2mpl.get_map('Set2','qualitative', 7)
colors = bmap.mpl_colors

params = {
'axes.labelsize': 20,
'font.size': 20,
'legend.fontsize': 15,
'xtick.labelsize': 20,
'ytick.labelsize': 20,
'text.usetex': True ,
'figure.figsize': [7, 5.5], # instead of 4.5, 4.5
'font.weight': 'bold',
'axes.labelweight': 'bold',
'ps.useafm' : True,
'pdf.use14corefonts':True,
'pdf.fonttype': 42,
'ps.fonttype': 42
 }
mpl.rcParams.update(params)
# In[data input with control noise]
# datafile=[r".\Cartpole\exp_s\exp_0\data.txt",
#           # r".\Cartpole\exp_s\exp_0.05u2\data.txt",
#           r".\Cartpole\exp_s\exp_0.2u\data.txt",
#           r".\Cartpole\exp_s\exp_0.4u\data.txt"
#         ]
# replanfile = r".\Cartpole\exp_s\d2c\perfcheck.txt"
# d2cfile = r".\Cartpole\exp_s\d2c\clopdata.txt"
# noise=[0,0.2,0.4]
# umax=12
datafile=[
            r".\Pendulum\exp_s\exp_0\data.txt",
            # r".\Pendulum\exp_s\exp_0.2u\data.txt",
            r".\Pendulum\exp_s\exp_0.4u\data.txt",
            r".\Pendulum\exp_s\exp_0.6u\data.txt"
        ]
replanfile = r".\Pendulum\exp_s\d2c\perfcheck.txt"
d2cfile = r".\Pendulum\exp_s\d2c\clopdata.txt"
noise=[0,0.4,0.6]
umax=5.8
# datafile=[r".\Swimmer3\exp_s\exp_0u1\data.txt",
#           r".\Swimmer3\exp_s\exp_0.2u\data.txt",
#           r".\Swimmer3\exp_s\exp_0.4u\data.txt"
#         ]
# replanfile = r".\Swimmer3\exp_s\d2c\perfcheck.txt"
# d2cfile = r".\Swimmer3\exp_s\d2c\clopdata.txt"
# noise=[0,0.2,0.4]
# umax=20
# In[with state + control noise]
# datafile=[
#             r".\Pendulum\exp_s\exp_0su1\data.txt",
#             r".\Pendulum\exp_s\exp_0.02su\data.txt",
#             r".\Pendulum\exp_s\exp_0.04su\data.txt",
#             r".\Pendulum\exp_s\exp_0.1su\data.txt"
#         ]
# replanfile = r".\Pendulum\exp_s\d2c\perfchecksu.txt"
# d2cfile = r".\Pendulum\exp_s\d2c\clopdatasu.txt"
# noise=[0,0.02,0.04,0.1]
# umax=5.8
# datafile=[r".\Cartpole\exp_s\exp_0su\data.txt",
#           r".\Cartpole\exp_s\exp_0.001su2\data.txt",
#           r".\Cartpole\exp_s\exp_0.004su\data.txt",
#           r".\Cartpole\exp_s\exp_0.02su\data.txt",
#         ]
# replanfile = r".\Cartpole\exp_s\d2c\perfchecksu.txt"
# d2cfile = r".\Cartpole\exp_s\d2c\clopdatasu.txt"
# noise=[0,0.001,0.004,0.02]
# umax=12
# datafile=[r".\Swimmer3\exp_s\exp_0su1\data.txt",
#           r".\Swimmer3\exp_s\exp_0.0003su\data.txt",
#           r".\Swimmer3\exp_s\exp_0.0006su\data.txt"
#         ]
# replanfile = r".\Swimmer3\exp_s\d2c\perfchecksu.txt"
# d2cfile = r".\Swimmer3\exp_s\d2c\clopdatasu.txt"
# noise=[0,0.0003,0.0006]
# umax=20
# In[with exploration noise]
# datafile=[
#             # r".\Pendulum\exp_e\exp_0.05\data.txt",
#             r".\Pendulum\exp_e\exp_0.5\data.txt",
#             # r".\Pendulum\exp_e\exp_1\data.txt",
#             r".\Pendulum\exp_e\exp_2.0\data.txt",
#             # r".\Pendulum\exp_e\exp_3\data.txt",
#             r".\Pendulum\exp_e\exp_6\data.txt"
#         ]
# replanfile = r".\Pendulum\exp_s\d2c\perfcheck.txt"
# d2cfile = r".\Pendulum\exp_s\d2c\clopdata.txt"
# # noise=[0.05,0.5,1,2.0,3]
# # noise=[0.5,2.0,6]
# noise=['$\sigma$=0.5','$\sigma$=2','$\sigma$=6']
# umax=5.8
# datafile=[
#           # r".\Cartpole\exp_e\exp_0.3\data.txt",
#           r".\Cartpole\exp_e\exp_0.3to0.01\data.txt",
#           r".\Cartpole\exp_e\exp_1to0.01\data.txt",
#           r".\Cartpole\exp_e\exp_4to0.01\data.txt"
#         ]
# replanfile = r".\Cartpole\exp_s\d2c\perfcheck.txt"
# d2cfile = r".\Cartpole\exp_s\d2c\clopdata.txt"
# # noise=['0.3','0.3to0.01','1to0.01']
# noise=['$\sigma$=0.3to0.01','$\sigma$=1to0.01','$\sigma$=4to0.01']
# umax=12
# In[plot]
curve_colors=[0,1,2,3,4,5,6]
curve_linestyle=['-','--','-.',':','--']
noise_legend=[]
noise_legend=["D2C Replan","D2C"]
for ns in noise:
    noise_legend.append("DDPG $\epsilon$="+format(100*ns,'.2f')+'\%')
    # noise_legend.append(ns)
plt.figure(1)

# replan
i, x, y = np.loadtxt(replanfile, dtype=np.float64, delimiter=' ',  unpack=True, usecols=(0,1,2))
i = 100*i
plt.plot(i,x,color='purple', linewidth=3)
#plt.plot(i, (x+y), alpha=0.3, color='orange')
#plt.plot(i, (x-y), alpha=0.3, color='orange')
plt.fill_between(i, (x+y), (x-y), alpha=0.2, color='purple')

# d2c
noise_range=100 # % of umax
y1=np.array(np.loadtxt(d2cfile))
perf1=np.mean(y1,axis=1)
cstd1=np.std(y1,axis=1)
step=noise_range/int(perf1.shape[0]-1)
sind=int(0/step)
eind=int(noise_range/step)+1
f6,=plt.plot(np.arange(sind,(eind-1)*step+0.001,step),perf1[sind:eind],color=colors[4], linewidth=3)
plt.fill_between(np.arange(sind,(eind-1)*step+0.001,step),(perf1[sind:eind]-cstd1[sind:eind]),(perf1[sind:eind]+cstd1[sind:eind]), alpha=0.3, color=colors[4])

plt.grid(axis='y', color='.910', linestyle='-', linewidth=1.5)
plt.grid(axis='x', color='.910', linestyle='-', linewidth=1.5)
for file,clr,lst in zip(datafile,curve_colors,curve_linestyle):
    i, x, y, z, a = np.loadtxt(file, dtype=np.float64, delimiter=',\t', unpack=True, usecols=(0,1,2,3,4))
    i = 100*i
    # plt.plot(i,-z,color=colors[clr], ls=lst, linewidth=3)
    # plt.fill_between(i, (-z+a), (-z-a), alpha=0.2, color=colors[clr])
    plt.plot(i,x,color=colors[clr], ls=lst, linewidth=3)
    plt.fill_between(i, (x+y), (x-y), alpha=0.2, color=colors[clr])

plt.legend(noise_legend,loc='upper left')
plt.xlabel('Std dev of process noise(\% of max. control)', fontweight='bold',fontsize=20)
plt.ylabel('L2 norm of terminal state error', fontweight='bold',fontsize=20)
# plt.ylabel('Episodic cost', fontweight='bold', fontsize=20)
# plt.ylim([1500,6000])
# plt.ylim([0,6])
# plt.xlim([0,12])
#print('averaged by {value1} rollouts'.format(value1=y.shape[1]))

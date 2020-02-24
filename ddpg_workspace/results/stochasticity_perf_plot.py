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
'axes.labelsize': 15,
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
# In[data input]
#datafile=[r"C:\Users\rwang\Desktop\DDPG_D2C\results\Cartpole\exp_0\data.txt",
#          r"C:\Users\rwang\Desktop\DDPG_D2C\results\Cartpole\exp_001\data.txt",
#          r"C:\Users\rwang\Desktop\DDPG_D2C\results\Cartpole\exp_005\data.txt"
#        ]
#noise=[0,0.01,0.05]
#umax=12
#datafile=[r"C:\Users\rwang\Desktop\DDPG_D2C\results\Pendulum\exp_0\data.txt",
#          r"C:\Users\rwang\Desktop\DDPG_D2C\results\Pendulum\exp_02\data.txt",
#          r"C:\Users\rwang\Desktop\DDPG_D2C\results\Pendulum\exp_05\data.txt"
#        ]
#noise=[0,0.2,0.5]
#umax=5.8
datafile=[r"C:\Users\rwang\Desktop\DDPG_D2C\results\Swimmer3\exp_0\data.txt",
          r"C:\Users\rwang\Desktop\DDPG_D2C\results\Swimmer3\exp_0005\data.txt"
        ]
noise=[0,0.005]
umax=20
# In[plot]
curve_colors=[0,1,2,3,4,5,6]
curve_linestyle=['-','--','-.',':']
noise_legend=[]
for ns in noise:
    noise_legend.append('$\epsilon=$'+format(ns/umax,'.4f'))
plt.figure(1)
plt.grid(axis='y', color='.910', linestyle='-', linewidth=1.5)
plt.grid(axis='x', color='.910', linestyle='-', linewidth=1.5)
for file,clr,lst in zip(datafile,curve_colors,curve_linestyle):
    i, x, y, z, a = np.loadtxt(file, dtype=np.float64, delimiter=',\t', unpack=True, usecols=(0,1,2,3,4))
    plt.plot(i/umax,-z,color=colors[clr], ls=lst, linewidth=3)
    plt.fill_between(i/umax, (-z+a), (-z-a), alpha=0.2, color=colors[clr])

plt.legend(noise_legend,loc='upper left')
plt.xlabel('Process noise level $\epsilon$ during testing', fontweight='bold',fontsize=20)
plt.ylabel('Averaged episodic cost', fontweight='bold', fontsize=20)
#print('averaged by {value1} rollouts'.format(value1=y.shape[1]))

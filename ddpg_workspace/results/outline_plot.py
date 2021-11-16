import numpy as np
import matplotlib.pyplot as plt
import brewer2mpl
from scipy import signal
from matplotlib import rcParams
import pandas as pd
import pylab

#plot preprocessing
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
rcParams.update(params)

data={'training time':[0.000146,0.000257,0.00479,0.00144,0.000442],'training variance':[0.000838,0.0025,0.014,float('nan'),float('nan')],'closed-loop variance':[float('nan'),0.52,float('nan'),0.95,0.234]}
#%% plot version 2
# df = pd.DataFrame(data)
# x = range(len(df.columns))
# dist=0.15
# rect = plt.bar(dist*(0-2), height=df.iat[0,0], width=0.1, color=colors[0], alpha=0.8,label="D2C pendulum")
# rect = plt.bar(dist*(1-2), height=df.iat[1,0], width=0.1, color=colors[1], alpha=0.8,label="D2C cartpole")
# rect = plt.bar(dist*(2-2), height=df.iat[2,0], width=0.1, color=colors[2], alpha=0.8,label="D2C 3-link swimmer")
# rect = plt.bar(dist*(3-2), height=df.iat[3,0], width=0.1, color=colors[3], alpha=0.8,label="D2C 6-link swimmer")
# rect = plt.bar(dist*(4-2), height=df.iat[4,0], width=0.1, color=colors[4], alpha=0.8,label="D2C fish")

# rect = plt.bar(1+dist*(0-1), height=df.iat[0,1], width=0.1, color=colors[0], alpha=0.8)
# rect = plt.bar(1+dist*(1-1), height=df.iat[1,1], width=0.1, color=colors[1], alpha=0.8)
# rect = plt.bar(1+dist*(2-1), height=df.iat[2,1], width=0.1, color=colors[2], alpha=0.8)

# rect = plt.bar(2+dist*(0-1), height=df.iat[1,2], width=0.1, color=colors[1], alpha=0.8)
# rect = plt.bar(2+dist*(1-1), height=df.iat[3,2], width=0.1, color=colors[3], alpha=0.8)
# rect = plt.bar(2+dist*(2-1), height=df.iat[4,2], width=0.1, color=colors[4], alpha=0.8)

# height=[0.00005,0.00007,0.0005,0.0002,0.0001]
# for i in range(0,5):
#     plt.text(dist*(i-2), df.iat[i,0]+height[i], str(df.iat[i,0]), ha="center", va="bottom",rotation=90,fontsize=11)

# height=[0.0001,0.0004,0.002,0.0002,0.0001]
# for i in range(0,3):
#     plt.text(dist*(i-1)+1, df.iat[i,1]+height[i], str(df.iat[i,1]), ha="center", va="bottom",rotation=90,fontsize=11)

# height=[0.1,0.3,0.2,0.5,0.15]
# xx=[0,1.85,0,2,2.15]
# for i in [1,3,4]:
#     plt.text(xx[i], df.iat[i,2]-height[i], str(df.iat[i,2]), ha="center", va="bottom",rotation=90,fontsize=11)

# plt.axhline(y=1, color='dodgerblue', linestyle='-',linewidth=3, label = "DDPG")
# plt.ylabel("D2C / DDPG fraction")
# pylab.yscale('log')
# plt.xticks(x,df.columns.values,rotation=10)
# plt.legend(loc = (0.01,0.7),fontsize=14,ncol=2)

#%% plot version 1
data={'training time':0.00142,'training variance':0.00578,'closed-loop variance':0.56}
df = pd.DataFrame(data,index=[0])
x = range(len(df.columns))
rect = plt.bar(x, height=df.loc[0,:], width=0.4, color='orange', alpha=0.6,label="D2C")

height=[0.0001,0.0003,0.01]
for i in x:
    plt.text(i, df.iat[0,i]+height[i], str(df.iat[0,i]), ha="center", va="bottom",rotation=90,fontsize=14)

plt.axhline(y=1, color='dodgerblue', linestyle='-',linewidth=3, label = "DDPG")
plt.ylabel("D2C / DDPG fraction")
pylab.yscale('log')
plt.xticks(x,df.columns.values,rotation=10)
plt.legend(loc = (0.01,0.75),fontsize=16)
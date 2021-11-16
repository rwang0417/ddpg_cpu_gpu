import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import brewer2mpl
import matplotlib as mpl

# In[plot settings]
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
mpl.rcParams.update(params)
# In[examples for process noise in u and states]
time_coef=1
data_slice_interval=1

# datafile=['.\Swimmer3\exp_s\exp_0u\ddpg_Swimmer3-v2_log.json',
#           '.\Swimmer3\exp_s\exp_0.0003su\ddpg_Swimmer3-v2_log.json',
#           '.\Swimmer3\exp_s\exp_0.0006su\ddpg_Swimmer3-v2_log.json'
#         ]
# data_slice_interval=1
# umax=20
#time_coef=31.067

# datafile=['.\Pendulum\exp_s\exp_0\ddpg_Pendulum-v2_log.json',
#           '.\Pendulum\exp_s\exp_0.02su\ddpg_Pendulum-v2_log.json',
#           '.\Pendulum\exp_s\exp_0.04su\ddpg_Pendulum-v2_log.json',
#           '.\Pendulum\exp_s\exp_0.5su\ddpg_Pendulum-v2_log.json'
#         ]
# data_slice_interval=150
# umax=5.8
# time_coef=0.2826

# datafile=['.\Cartpole\exp_s\exp_0\ddpg_Cartpole-v2_log.json',
#           '.\Cartpole\exp_s\exp_0.001su2\ddpg_Cartpole-v2_log.json',
#           '.\Cartpole\exp_s\exp_0.004su\ddpg_Cartpole-v2_log.json',
#           '.\Cartpole\exp_s\exp_0.02su\ddpg_Cartpole-v2_log.json'
#         ]
# data_slice_interval=250
# umax=12
# time_coef=0.378
# In[examples for process noise in u]
# datafile=['.\Swimmer3\exp_s\exp_0u1\ddpg_Swimmer3-v2_log.json',
#           '.\Swimmer3\exp_s\exp_0.2u\ddpg_Swimmer3-v2_log.json',
#           '.\Swimmer3\exp_s\exp_0.4u\ddpg_Swimmer3-v2_log.json'
#         ]
# data_slice_interval=1
# umax=20
#time_coef=31.067

# datafile=['.\Pendulum\exp_s\exp_0\ddpg_Pendulum-v2_log.json',
#           '.\Pendulum\exp_s\exp_0.2u\ddpg_Pendulum-v2_log.json',
#           '.\Pendulum\exp_s\exp_0.4u\ddpg_Pendulum-v2_log.json',
#           '.\Pendulum\exp_s\exp_0.6u\ddpg_Pendulum-v2_log.json',
#           # '.\Pendulum\exp_s\exp_0.1su\ddpg_Pendulum-v2_log.json',
#         ]
# data_slice_interval=150
# umax=5.8
# time_coef=0.2826

# datafile=['.\Cartpole\exp_s\exp_0\ddpg_Cartpole-v2_log.json',
#           '.\Cartpole\exp_s\exp_0.05u\ddpg_Cartpole-v2_log.json',
#           '.\Cartpole\exp_s\exp_0.2u\ddpg_Cartpole-v2_log.json',
#           '.\Cartpole\exp_s\exp_0.4u\ddpg_Cartpole-v2_log.json'
#         ]
# data_slice_interval=250
# umax=12
# time_coef=0.378
# In[for sigma]
time_coef=1
sigma=[]

#datafile=['.\Swimmer3\exp_0\ddpg_Swimmer3-v2_log.json',
#          '.\Swimmer3\exp_001u\ddpg_Swimmer3-v2_log.json',
#          '.\Swimmer3\exp_01u\ddpg_Swimmer3-v2_log.json'
#        ]
#data_slice_interval=1
#umax=20
#time_coef=31.067

# datafile=['.\Pendulum\exp_e\exp_0.05\ddpg_Pendulum-v2_log.json',
#           '.\Pendulum\exp_e\exp_0.5\ddpg_Pendulum-v2_log.json',
#           '.\Pendulum\exp_e\exp_1\ddpg_Pendulum-v2_log.json',
#           '.\Pendulum\exp_e\exp_2.0\ddpg_Pendulum-v2_log.json',
#           '.\Pendulum\exp_e\exp_3.0\ddpg_Pendulum-v2_log.json'
#         ]
# data_slice_interval=150
# umax=5.8
# sigma=[0.05,0.5,1,2.0,3]
#time_coef=0.2826

datafile=['.\Cartpole\exp_e\exp_0.3\ddpg_Cartpole-v2_log.json',
          '.\Cartpole\exp_e\exp_0.3to0.01\ddpg_Cartpole-v2_log.json',
          '.\Cartpole\exp_e\exp_1to0.01\ddpg_Cartpole-v2_log.json'
        ]
data_slice_interval=250
umax=12
sigma=['0.3','0.3to0.01','1to0.01']
#time_coef=0.378
# In[data input]
data=[]
timelist=[]
episode_reward_fraction=[]
episode_reward=[]
loss=[]
process_noise_std=[]
theta=[]
for file in datafile:
    with open(file, 'r') as f:
        data.append(json.load(f))
    episode_reward_orig = np.array(data[-1]['episode_reward'])
    loss_orig=np.array([ld for ld in data[-1]['loss'] if ld == ld])
    loss.append(loss_orig)
    episode_reward_fraction.append(episode_reward_orig/episode_reward_orig[-1])
    episode_reward.append(-episode_reward_orig)
    timelist.append(data[-1]['episode'])
    process_noise_std.append(data[-1]['process_noise_std'])
#    theta.append(data[-1]['theta'])
#    if len(episode_reward_orig) != len(data[0]['episode_reward']):
#        print('ERROR: episode doesn\'t have the same length...')
# In[adjust the longer training data length to make better plot]
#episode_reward[2]=episode_reward[2][:1750:1]
#timelist[2] = timelist[2][:1750:1]
#episode_reward[3]=episode_reward[3][:30001:1]
#timelist[3] = timelist[3][:30001:1]
# In[plot]
b, a=signal.butter(8, 0.025)
curve_colors=[0,1,2,3,4,5,6]
curve_linestyle=[':','--','-.','-','-.']
process_noise_std_flat = [item for sublist in process_noise_std for item in sublist]
theta_flat = [item for sublist in theta for item in sublist]
sigma_flat = sigma
noise_legend=[]
theta_legend=[]
sigma_legend=[]
for m in process_noise_std_flat:
    m='$\epsilon=$'+format(m/umax,'.3f')
    noise_legend.append(m)
for m in sigma_flat:
    # m='$\sigma=$'+format(m,'.2f')
    m='$\sigma=$'+m
    sigma_legend.append(m)
plt.figure(1)
for time,erf,clr,lst in zip(timelist,episode_reward,curve_colors,curve_linestyle):
#    plt.plot(time, erf, color=colors[1], alpha=0.9)
    signal_raw=signal.filtfilt(b, a, erf)
    signal_slice=signal_raw[::data_slice_interval]
    time_slice=time[::data_slice_interval]
    time_slice_time=[x*time_coef for x in time_slice]
    # plt.plot(time_slice[len(time_slice)-len(signal_slice)-1:-1], signal_slice, color=colors[clr], ls=lst,linewidth=3)
    plt.plot(time_slice, signal_slice, color=colors[clr], ls=lst,linewidth=3)
plt.grid(axis='y', color='.910', linestyle='-', linewidth=1.5)
plt.grid(axis='x', color='.910', linestyle='-', linewidth=1.5)
plt.xlabel('Num of rollouts', fontsize=20)
plt.ylabel('Episodic cost', fontsize=20)
# plt.ylabel('Loss', fontsize=20)
plt.legend(sigma_legend)
# plt.legend(noise_legend)
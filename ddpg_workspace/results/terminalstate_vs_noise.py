import numpy as np
import matplotlib.pyplot as plt
import brewer2mpl
from scipy import signal
from matplotlib import rcParams
import pandas as pd

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
ddpg = {}

#%% pendulum
log_filename_pre = './Pendulum/exp_e/'
sigma=[0.05,0.5,1,2.0,3]
ddpg_tse=[]
for i in sigma:
    y=np.array(np.loadtxt(log_filename_pre+'exp_'+'{}'.format(i)+'/nominal_state.txt'))
    ddpg_tse.append(np.sqrt(y[-1][0]**2+y[-1][1]**2))
ddpg['pendulum'] = ddpg_tse
print('pendulum ddpg variance = {}'.format(np.std(ddpg_tse)))

#%% cartpole
# log_filename_pre = './Cartpole/exp_e/'
# sigma=['0.3','0.3to0.01','1to0.01']
# ddpg_tse=[]
# for i in sigma:
#     y=np.array(np.loadtxt(log_filename_pre+'exp_'+i+'/nominal_state.txt'))
#     ddpg_tse.append(np.sqrt(y[-1][0]**2+(y[-1][1])**2+y[-1][2]**2+y[-1][3]**2))
# ddpg['cartpole'] = ddpg_tse
# print('cartpole ddpg variance = {}'.format(np.std(ddpg_tse)))

#%% swimmer3
# log_filename_pre = './Swimmer3/'
# ddpg_exp=[]
# d2c_exp=[]
# for i in range(1,5):
#     y=np.array(np.loadtxt(log_filename_pre+ddpg_filename_exp+'{}'.format(i)+'/s3_nominal_state.txt'))
#     ddpg_exp.append(np.sqrt((y[-1][0]-0.6)**2+(y[-1][1]+0.5)**2))
# for i in range(1,5):
#     y=np.array(np.loadtxt(log_filename_pre+d2c_filename_exp+'/state0{}.txt'.format(i)))
#     d2c_exp.append(np.sqrt((y[-1][0]-0.5)**2+(y[-1][1]+0.5)**2))
# d2c['swimmer3'] = d2c_exp
# ddpg['swimmer3'] = ddpg_exp
# print('swimmer3 d2c variance = {}'.format(np.std(d2c_exp)))
# print('swimmer3 ddpg variance = {}'.format(np.std(ddpg_exp)))

#%% swimmer6
# log_filename_pre = './Swimmer6/'
# d2c_exp=[]
# for i in range(1,5):
#     y=np.array(np.loadtxt(log_filename_pre+d2c_filename_exp+'/state0{}.txt'.format(i)))
#     d2c_exp.append(np.sqrt((y[-1][0]-0.5)**2+(y[-1][1]+0.6)**2))
# d2c['swimmer6'] = d2c_exp
# print('swimmer6 d2c variance = {}'.format(np.std(d2c_exp)))

#%% fish
# log_filename_pre = './Fish/'
# d2c_exp=[]
# for i in range(1,5):
#     y=np.array(np.loadtxt(log_filename_pre+d2c_filename_exp+'/state0{}.txt'.format(i)))
#     d2c_exp.append(np.sqrt((y[-1][0])**2+(y[-1][1]-0.4)**2+(y[-1][2]-0.2)**2)+(y[-1][3]-1)**2)
# d2c['fish'] = d2c_exp
# print('fish d2c variance = {}'.format(np.std(d2c_exp)))

#%% material
# log_filename_pre = './Material/'
# d2c_exp=np.load(log_filename_pre+'end_disc.npy')
# # for i in range(1,5):
# #     y=np.array(np.loadtxt(log_filename_pre+d2c_filename_exp+'/state0{}.txt'.format(i)))
# #     d2c_exp.append(np.sqrt((y[-1][0])**2+(y[-1][1]-0.4)**2+(y[-1][2]-0.2)**2)+(y[-1][3]-1)**2)
# d2c['material'] = d2c_exp[0:4]
# print('material d2c variance = {}'.format(np.std(d2c_exp)))

#%% plot
sigma_legend=[]
for m in sigma:
    m='$\sigma=$'+format(m,'.2f')
    # m='$\sigma=$'+m
    sigma_legend.append(m)
plt.plot(range(len(sigma)),ddpg_tse,'r.')
plt.xticks(range(len(sigma)),sigma,rotation=0)
plt.xlabel('Exploration noise $\sigma$', fontsize=20)
plt.ylabel('Terminal state error', fontsize=20)
# ddpg_df = pd.DataFrame(ddpg)
# fig = plt.figure(figsize = (7, 5.5))
# ax1 = fig.add_subplot(1, 1, 1)
# ax1.plot(d2c_df.columns.values, d2c_df.loc[0, :], 'r.', label = "D2C")
# ax1.plot(ddpg_df.columns.values, ddpg_df.loc[0, :], 'g.', label = "DDPG")
# for i in range(1,3):
#     ax1.plot(d2c_df.columns.values, d2c_df.loc[i, :], 'r.')
#     ax1.plot(ddpg_df.columns.values, ddpg_df.loc[i, :], 'g.')

# # ax1.set_xlabel("model")
# # ax1.set_title("Variance comparison D2C vs. DDPG")
# ax1.set_ylabel("final distance to target")
# plt.xticks(d2c_df.columns.values,d2c_df.columns.values,rotation=30)
# ax1.legend(loc = "best")

# box1=plt.boxplot((d2c_df.iloc[:, 0], d2c_df.iloc[:, 1], d2c_df.iloc[:, 2]))
# for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
#         plt.setp(box1[item], color=colors[1])
# box2=plt.boxplot((ddpg_df.iloc[:, 0], ddpg_df.iloc[:, 1], ddpg_df.iloc[:, 2]))
# for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
#         plt.setp(box2[item], color=colors[3])
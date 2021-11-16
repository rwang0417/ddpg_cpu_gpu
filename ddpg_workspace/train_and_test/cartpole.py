import numpy as np
import gym
import time
import os

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, BatchNormalization, Lambda
from keras.optimizers import Adam
from keras import initializers
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import FileLogger, ModelIntervalCheckpoint, TestLogger

import common_func
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
ENV_NAME = 'Cartpole-v2'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
# np.random.seed(123)
# env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Next, we build a very simple model.
state_input = Input(shape=(1,) + env.observation_space.shape, name='state_input')
flattened_state = Flatten()(state_input)
y = Dense(400)(flattened_state)
y = Activation('relu')(y)
y = Dense(300)(y)
y = Activation('relu')(y)
y = Dense(nb_actions, kernel_initializer=initializers.RandomUniform(minval=-0.003,  maxval=0.003))(y)
y = Activation('tanh')(y)
y = Lambda(lambda x: x * 12)(y)


actor = Model(inputs=[state_input], outputs=y)
print(actor.summary())

# build the critic model
action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(400)(x)
x = Activation('relu')(x)
x = Dense(300)(x)
x = Activation('relu')(x)
x = Dense(1, kernel_initializer=initializers.RandomUniform(minval=0.0003, maxval=0.0003))(x)
x = Activation('linear')(x)

critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

filename_exp = 'exp_e/exp_4to0.01'
log_filename_pre = '../results/Cartpole/'
process_noise_std = 0.45*12
theta=0.15
sigma=4
sigma_min=0.01

GAMMA = 0.99    # GAMMA of our cumulative reward function
STEPS_PER_EPISODE = 30     # No. of time-steps per episode

# configure and compile our agent by using built-in Keras optimizers and the metrics!
# allocate the memory by specifying the maximum no. of samples to store
memory = SequentialMemory(limit=600000, window_length=1)
# random process for exploration noise
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=theta, dt=0.1, mu=0., sigma=sigma,sigma_min=sigma_min)
# define the DDPG agent
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=GAMMA, target_model_update=1e-3)
# compile the model
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mse'])

callbacks = common_func.build_callbacks(ENV_NAME, log_filename_pre, filename_exp)
# ----------------------------------------------------------------------------------------------------------------------------------------
# Training phase
# fitting the agent, after training is done, we save the final weights.
# 500000

# agent.fit(env, nb_steps=600000, visualize=False, callbacks=callbacks, verbose=1, gamma=GAMMA, nb_max_episode_steps=STEPS_PER_EPISODE, process_noise_std=process_noise_std)
# agent.save_weights(log_filename_pre+filename_exp+'/ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
# common_func.save_process_noise(ENV_NAME, log_filename_pre, filename_exp, process_noise_std, theta)

# -----------------------------------------------------------------------------------------------------------------------------------------
# Testing phase

agent.load_weights(log_filename_pre+filename_exp +'/ddpg_{}_weights.h5f'.format(ENV_NAME))
history, state_history_nominal, episode_reward_nominal, action_history = agent.test(env, nb_episodes=1, visualize=True, action_repetition=1, nb_max_episode_steps=STEPS_PER_EPISODE, \
                                                         initial_state=[0, np.pi, 0, 0], std_dev_noise=0, gamma=GAMMA, process_noise_std=process_noise_std)

# np.savetxt(log_filename_pre+filename_exp+'/nominal_action.txt', action_history)
# np.savetxt(log_filename_pre+filename_exp+'/nominal_state.txt', state_history_nominal)

# print(action_history,state_history_nominal,episode_reward_nominal)
print(state_history_nominal[-1],np.linalg.norm(state_history_nominal[-1], axis=0))

# -----------------------------------------------------------------------------------------------------------------------------------------

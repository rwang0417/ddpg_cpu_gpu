import numpy as np
import sys
sys.path.insert(1, '../train_and_test/')

from pendulum import agent 
from  pendulum import env, GAMMA, STEPS_PER_EPISODE, filename_exp, log_filename_pre

def frange(start, stop, step):
    i = start
    while i < stop:
        yield round(i,2)
        i += step


u_max = 5.8
n_samples = 400 # number of monte carlo samples

path_to_save = log_filename_pre+filename_exp+"/data.txt"
f = open(path_to_save, "a")


for i in frange(0.0, 0.120001, 0.02):

    episode_reward_n = 0
    Var_n = 0
    terminal_mse = 0
    Var_terminal_mse = 0

    for j in range(n_samples):

        history, state_history, episode_reward, action_history = agent.test(env, nb_episodes=1, visualize=False, action_repetition=1, nb_max_episode_steps=STEPS_PER_EPISODE, initial_state=[np.pi, 0], process_noise_std=i*u_max, gamma=GAMMA)
        # print(state_history[-1])
        episode_reward_n += episode_reward
        Var_n += (episode_reward)**2
        terminal_mse += np.linalg.norm(state_history[STEPS_PER_EPISODE], axis=0)
        Var_terminal_mse += (np.linalg.norm(state_history[STEPS_PER_EPISODE], axis=0))**2

    terminal_mse_avg = terminal_mse/n_samples
    episode_reward_n_avg = episode_reward_n/n_samples
    var_avg = (1/(n_samples*(n_samples-1)))*(n_samples*Var_n - episode_reward_n**2)
    Var_terminal_mse_avg = (1/(n_samples*(n_samples-1)))*(n_samples*Var_terminal_mse - terminal_mse**2)

    if var_avg > 0 :
        std_dev_avg = np.sqrt(var_avg)

    else:

        std_dev_avg = 0
    std_dev_mse = np.sqrt(Var_terminal_mse_avg)

    f.write(str(i)+",\t"+str(terminal_mse_avg)+",\t"+str(std_dev_mse)+",\t"+str(episode_reward_n_avg)+",\t"+str(std_dev_avg)+"\n")
    print(terminal_mse_avg, std_dev_mse, episode_reward_n_avg, std_dev_avg)
f.close()



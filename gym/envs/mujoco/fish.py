import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class FishEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):

        mujoco_env.MujocoEnv.__init__(self, 'fish_old.xml', 5, np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0]), np.zeros((13,)))
        utils.EzPickle.__init__(self)

    def step(self, a,process_noise_std=0):

        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0:3]
        self.do_simulation(a, self.frame_skip, process_noise_std)

        reward_fwd = -15*((self.sim.data.qpos[0])**2 + (self.sim.data.qpos[1]-0.4)**2 + (self.sim.data.qpos[2] - 0.2)**2) -\
                     .5*(np.square(self.sim.data.qpos[4:7]).sum() + (self.sim.data.qpos[3]-1)**2) #(np.square(self.sim.data.qpos[7:14]).sum())
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        #print(a,"\n")
        reward = (reward_fwd + reward_ctrl)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        return np.concatenate([qpos.flat, qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos, #+ self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel #+ self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()

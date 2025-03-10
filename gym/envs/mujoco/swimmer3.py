import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class Swimmer3Env(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'swimmer3.xml', 1, np.array([0.0,0.0,0,0,0]), np.array([0,0,0,0,0]))
        utils.EzPickle.__init__(self)

    def step(self, a, process_noise_std=0):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip, process_noise_std)
        xposafter = self.sim.data.qpos[0]
        yposafter = self.sim.data.qpos[1]
        #print(a[0],'\t',a[1])
        #print("x of the head: ", xposafter, "y of the head: ", yposafter, "angles", self.sim.data.qpos[2:5])
        reward_fwd = -80*((xposafter-0.6)**2+ (yposafter+0.5)**2) #(xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        #print(ob)
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

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

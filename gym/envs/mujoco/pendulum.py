import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import time

class PendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'pendulum.xml', 1, np.array([np.pi]), np.array([0]))

    def step(self, a, process_noise_std=0):
        self.do_simulation(a, self.frame_skip, process_noise_std)
        ob = self._get_obs()
        reward = -10*(2*ob[0]**2 + ob[1]**2) - 3*a[0]**2
        notdone = np.isfinite(ob).all() #and #and (np.abs(ob[1]) <= .2)
        done = not notdone
        
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos #+ self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel #+ self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.angle_normalize(self.sim.data.qpos), self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    def angle_normalize(self,x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)

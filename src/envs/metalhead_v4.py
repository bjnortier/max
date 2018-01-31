import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class MetalheadEnvV4(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), 'assets', 'metalhead_v4.xml')
        mujoco_env.MujocoEnv.__init__(self, model_path, 5)
        utils.EzPickle.__init__(self)

    def _step(self, action):
        xposbefore = self.model.data.qpos[0, 0]
        torsoanglebefore = self.model.data.qpos[2, 0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.model.data.qpos[0, 0]
        torso_pitch_angle_after = self.model.data.qpos[2, 0]
        torso_roll_angle_after = self.model.data.qpos[3, 0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward_torso_pitch = -0.02 * np.square(torso_pitch_angle_after)
        reward_torso_roll = -0.02 * np.square(torso_roll_angle_after)
        reward = reward_ctrl + reward_run + reward_torso_pitch + reward_torso_roll
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            self.model.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = 5

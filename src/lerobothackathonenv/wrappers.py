import gymnasium as gym
import numpy as np
from typing import Any, Dict, List
from .trajectory_set import TrajectorySet


class ReferenceStateInitializationWrapper(gym.Wrapper):
    def __init__(self, env, trajectory_set: TrajectorySet):
        super().__init__(env)
        self.trajectory_set = trajectory_set
        self.qpos_array = None
        self.qvel_array = None
        self._load_states()

    def _load_states(self):
        all_qpos = []
        all_qvel = []
        for traj_idx in range(self.trajectory_set.index):
            trajectory_data = self.trajectory_set.get_trajectory(traj_idx)
            simulator_states = trajectory_data['simulator_states']
            for state in simulator_states:
                all_qpos.append(state.qpos)
                all_qvel.append(state.qvel)
        self.qpos_array = np.array(all_qpos)
        self.qvel_array = np.array(all_qvel)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if len(self.qpos_array) > 0:
            random_idx = np.random.randint(0, len(self.qpos_array))
            self.env.unwrapped.dm_control_env.physics.data.qpos[:] = self.qpos_array[random_idx]
            self.env.unwrapped.dm_control_env.physics.data.qvel[:] = self.qvel_array[random_idx]
            self.env.unwrapped.dm_control_env.physics.forward()

        return obs, info

    def step(self, action):
        return self.env.step(action)

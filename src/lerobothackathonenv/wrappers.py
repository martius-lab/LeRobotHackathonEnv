import gymnasium as gym
import numpy as np
from typing import Any, Dict, List
from .trajectory_set import TrajectorySet


class ReferenceStateInitializationWrapper(gym.Wrapper):
    def __init__(self, env, trajectory_set: TrajectorySet):
        super().__init__(env)
        self.trajectory_set = trajectory_set

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.trajectory_set.load_into_cache()

        if self.trajectory_set.cache:
            sampled_state = self.trajectory_set.sample_state(1)[0]

            self.env.unwrapped.dm_control_env.physics.data.qpos[:] = sampled_state["qpos"]
            self.env.unwrapped.dm_control_env.physics.data.qvel[:] = sampled_state["qvel"]
            self.env.unwrapped.dm_control_env.physics.forward()

        return sampled_state, info

    def step(self, action):
        return self.env.step(action)

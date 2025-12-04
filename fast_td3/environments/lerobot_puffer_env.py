import numpy as np
import torch

import gymnasium as gym
import pufferlib.vector as pv
from gymnasium.spaces import Box
from pufferlib.emulation import GymnasiumPufferEnv


class LeRobotPufferEnv:
    """
    PufferLib-backed vector env wrapper for LeRobot Gymnasium envs.

    Exposes the interface expected by FastTD3's training loop:
    - Attributes: num_envs, num_obs, num_actions, asymmetric_obs, max_episode_steps.
    - Methods: reset(), step(actions) returning torch tensors and infos dict.
    """

    def __init__(
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        device: torch.device,
        max_episode_steps: int = 1000,
    ):
        if num_envs <= 0:
            raise ValueError(f"num_envs must be > 0, got {num_envs}")

        # Ensure LeRobot gym env ids are registered (LeRobot-v0, LeRobotGoalConditioned-v0, ...)
        import lerobothackathonenv  # noqa: F401

        self.device = device
        self.max_episode_steps = max_episode_steps

        # Minimal PufferLib setup, mirroring the README style
        num_workers = min(4, num_envs)
        def make_env():
            env = gym.make(env_id, max_episode_steps=max_episode_steps)
            return env


        if num_workers == 1:
            # Single-environment setup with an explicit episode horizon.
            self.vec_env = GymnasiumPufferEnv(
                env=gym.make(env_id, max_episode_steps=max_episode_steps),
                seed=seed,
            )
            obs_space = self.vec_env.observation_space
            act_space = self.vec_env.action_space
        else:
            self.vec_env = pv.make(
                GymnasiumPufferEnv,
                env_args=None,
                env_kwargs={"env_creator": make_env},
                backend=pv.Multiprocessing,
                num_envs=num_envs,
                num_workers=num_workers,
                batch_size=num_envs,
                seed=seed,
            )
            obs_space = self.vec_env.single_observation_space
            act_space = self.vec_env.single_action_space

        self.num_envs = num_envs
        self.asymmetric_obs = False
        self.num_obs = obs_space.shape[0]
        self.num_actions = act_space.shape[0]

    def reset(self) -> torch.Tensor:
        obs, _ = self.vec_env.reset()
        return torch.as_tensor(obs, device=self.device, dtype=torch.float32)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if not isinstance(actions, torch.Tensor):
            raise TypeError(f"Expected actions as torch.Tensor, got {type(actions)}")

        obs, rewards, terminals, truncations, infos = self.vec_env.step(
            actions.detach().cpu().numpy()
        )

        obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        rewards_t = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        dones_np = np.logical_or(terminals, truncations)
        dones_t = torch.as_tensor(dones_np, device=self.device, dtype=torch.long)
        trunc_t = torch.as_tensor(truncations, device=self.device, dtype=torch.long)

        info_ret = {
            "time_outs": trunc_t,
            "observations": {"raw": {"obs": obs_t.clone()}},
        }
        if isinstance(infos, list):
            success_np = np.asarray([i["success"] for i in infos], dtype=np.float32)
        else:
            success_np = np.asarray(infos["success"], dtype=np.float32)
        success_t = torch.as_tensor(
            success_np, device=self.device, dtype=torch.float32
        )
        info_ret["success"] = success_t

        return obs_t, rewards_t, dones_t, info_ret

    def close(self):
        self.vec_env.close()
    
    def render(self) -> np.ndarray:
        assert (
            self.num_envs == 1
        ), "Currently only supports single environment rendering"
        return self.vec_env.render()

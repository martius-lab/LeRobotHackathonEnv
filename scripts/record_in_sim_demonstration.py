from typing import Dict, List
from dataclasses import dataclass
import time
import numpy as np
import select
import sys
import tty
import termios
import os
import glob
import fcntl
import contextlib
from collections import deque

import lerobothackathonenv as _
from lerobothackathonenv.env import LeRobot
from lerobothackathonenv.structs import MujocoState
import gymnasium as gym
from numpy.typing import NDArray

from lerobot.configs import parser
from trajectory_set import TrajectorySet
from lerobot.teleoperators import (
    TeleoperatorConfig,
    make_teleoperator_from_config,
    so101_leader,
)
env: LeRobot = gym.make("LeRobotGoalConditioned-v0")
env.unwrapped.render_to_window()
observation, info = env.reset()

class RateLimiter:
    def __init__(self, calls_per_second):
        self.min_interval = 1.0 / (calls_per_second + 1)
        self.last_call_time = 0.0

    def wait(self):
        current_time = time.time()
        next_allowed_time = self.last_call_time + self.min_interval
        wait_time = next_allowed_time - current_time
        if wait_time > 0:
            time.sleep(wait_time)
            current_time = time.time()
        self.last_call_time = current_time

class TerminalRewardPlotter:
    def __init__(self, window_size=50, bar_width=60):
        self.window_size = window_size
        self.bar_width = bar_width
        self.rewards = deque(maxlen=window_size)
        self.min_reward = 0  # Will be updated as we see rewards
        self.max_reward = 0  # Best reward is 0
        
    def add_reward(self, reward):
        self.rewards.append(reward)
        # Update min reward (worst we've seen)
        if reward < self.min_reward:
            self.min_reward = reward
    
    def get_bar_representation(self, reward):
        """Convert reward to horizontal bar representation"""
        if self.min_reward == self.max_reward:
            # All rewards are the same, show full bar
            filled_length = self.bar_width // 2
        else:
            # Normalize reward to [0, 1] where 1 is best (0) and 0 is worst
            normalized = (reward - self.min_reward) / (self.max_reward - self.min_reward)
            filled_length = int(normalized * self.bar_width)
        
        # Create bar with █ for filled and ░ for empty
        bar = '█' * filled_length + '░' * (self.bar_width - filled_length)
        return bar
    
    def print_current_reward(self, reward):
        """Print current reward with bar visualization"""
        bar = self.get_bar_representation(reward)
        print(f"Reward: {reward:7.3f} |{bar}| (Best: 0.000, Worst: {self.min_reward:7.3f})")
    
    def print_recent_rewards(self):
        """Print recent rewards as a mini chart"""
        if len(self.rewards) < 2:
            return
            
        print("\nRecent rewards:")
        for i, reward in enumerate(list(self.rewards)[-10:]):  # Show last 10
            bar = self.get_bar_representation(reward)
            step_num = len(self.rewards) - 10 + i if len(self.rewards) >= 10 else i
            print(f"  {step_num:3d}: {reward:7.3f} |{bar[:30]}|")  # Shorter bars for history
    
    def reset(self):
        """Reset the plotter for new trajectory"""
        self.rewards.clear()
        self.min_reward = 0
        print("\n" + "="*80)
        print("New trajectory started - reward history cleared")
        print("="*80)

def proper_array_str(array: NDArray):
    return " ".join(
        f"{v:7.3f}{'\n' + 23 * " " if (i + 1) % 7 == 0 else ''}"
        for i, v in enumerate(array)
    )

def teleop_action2env_action(teleop_action: Dict[str, float]) -> NDArray:
    keys = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos"
    ]
    vector = np.array([teleop_action[k] for k in keys])
    vector *= 2 * np.pi / 360
    return vector

def check_keyboard_input():
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.read(1)
    return None

@dataclass
class TeleoperateConfig:
    teleop: TeleoperatorConfig
    fps: int = 60
    teleop_time_s: float | None = None
    display_data: bool = False

ts = TrajectorySet()
reward_plotter = TerminalRewardPlotter(window_size=50, bar_width=60)
observations, rewards, terminations, trunctuations, infos, simulator_states = (list() for i in range(6))

@parser.wrap()
def run_collection_loop(cfg: TeleoperateConfig):
    global observations, rewards, terminations, trunctuations, infos, simulator_states
    print(cfg)
    print("Press 'r' to reset the environment")
    print("Press 'd' to discard current trajectory and reset")

    # Set terminal to non-blocking mode
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())

        teleop = make_teleoperator_from_config(cfg.teleop)
        teleop.connect()
        rl = RateLimiter(20)
        t = time.time()

        while True:
            # Check for keyboard input
            key = check_keyboard_input()
            if key == 'r':
                print("Resetting environment...")
                observation, info = env.reset()
                ts.record_trajectory(observations, rewards, terminations, trunctuations, infos, simulator_states)
                observations, rewards, terminations, trunctuations, infos, simulator_states = (list() for i in range(6))
                simulator_states.append(env.sim_state)
                reward_plotter.reset()
                continue
            elif key == 'd':
                print("Discarding current trajectory and resetting...")
                observation, info = env.reset()
                observations, rewards, terminations, trunctuations, infos, simulator_states = (list() for i in range(6))
                simulator_states.append(env.sim_state)
                reward_plotter.reset()
                continue

            teleop_action = teleop.get_action()
            action = teleop_action2env_action(teleop_action)

            observation, reward, terminated, trunctuated, info = env.step(action)
            observations.append(observation)
            rewards.append(reward)
            terminations.append(terminated)
            trunctuations.append(trunctuated)
            infos.append(info)
            simulator_states.append(env.sim_state)

            # Add reward to terminal plotter and display
            reward_plotter.add_reward(reward)
            reward_plotter.print_current_reward(reward)
            
            done = terminated or trunctuated
            if done:
                reward_plotter.print_recent_rewards()
                ts.record_trajectory(observations, rewards, terminations, trunctuations, infos, simulator_states)
                observations, rewards, terminations, trunctuations, infos, simulator_states = (list() for i in range(6))
                observation, info = env.reset()
                simulator_states.append(env.sim_state)
                reward_plotter.reset()
            env.unwrapped.render_to_window()
            rl.wait()
            t = time.time()

    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
    run_collection_loop()

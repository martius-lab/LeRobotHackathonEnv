from typing import Dict
from dataclasses import dataclass
import time
import numpy as np
import select
import sys
import tty
import termios

import lerobothackathonenv as _
from lerobothackathonenv.env import LeRobot
import gymnasium as gym
from numpy.typing import NDArray

from lerobot.configs import parser
from lerobot.teleoperators import (
    TeleoperatorConfig,
    make_teleoperator_from_config,
    so101_leader,
)

env: LeRobot = gym.make("LeRobotGoalConditioned-v0")
env.unwrapped.render_to_window()
obs, info = env.reset()

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

def proper_array_str(array: NDArray):
    return " ".join(
        f"{v:7.3f}{'\n' + 23 * " " if (i + 1) % 7 == 0 else ''}"
        for i, v in enumerate(array)
    )

def print_obs(obs: dict):
    print("")
    print(f"=== observation dict: ===")
    for k, v in obs.items():
        print(f"{k:20} is {proper_array_str(v):10}")

def print_reward(reward: float):
    print(f"~~~ Reward: {reward: 10.3f} ~~~")

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

@parser.wrap()
def run_collection_loop(cfg: TeleoperateConfig):
    print(cfg)
    print("Press 'r' to reset the environment")
    
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
                obs, info = env.reset()
                print_obs(obs)
                continue
            
            teleop_action = teleop.get_action()
            action = teleop_action2env_action(teleop_action)
            print(teleop_action, action)

            observation, reward, terminated, trunctuated, info = env.step(action)
            done = terminated or trunctuated
            if done:
                observation, info = env.reset()
            print_obs(observation)
            print_reward(reward)
            env.unwrapped.render_to_window()
            rl.wait()
            print(1 / (time.time() - t))
            t = time.time()
    
    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
    run_collection_loop()

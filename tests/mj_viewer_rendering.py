from time import sleep
from lerobothackathonenv.env import LeRobot
import gymnasium as gym
import lerobothackathonenv as _
from numpy.typing import NDArray

env: LeRobot = gym.make("LeRobotGoalConditioned-v0")
env.unwrapped.render_to_window()

obs = env.reset()

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

while True:
    action = env.action_space.sample()
    observation, reward, terminated, trunctuated, info = env.step(action)
    print_obs(observation)
    print_reward(reward)
    env.unwrapped.render_to_window()

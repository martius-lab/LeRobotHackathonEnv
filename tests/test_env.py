from lerobothackathonenv.env import LeRobot
from lerobothackathonenv.structs import MujocoState
from numpy import ndarray

import gymnasium as gym
env: LeRobot = gym.make("LeRobot-v0")

def test_reset():
    # ~ see whether reset function works
    obs, info = env.reset()

def test_obs():
    obs, info = env.reset()
    keys = set(obs.keys())
    # ~ test dictionary keys
    assert "qpos" in keys
    assert "qvel" in keys
    assert "actuator_force" in keys
    assert "gripper_pos" in keys
    # ~ test dtype
    for key in keys:
        assert type(obs[key]) == ndarray

def test_step():
    obs, info = env.reset()
    action = env.action_space.sample()
    # ~ test step function
    result = env.step(action)
    assert len(result) == 5

def test_rendering():
    image = env.unwrapped.render(width=640, height=480)
    assert image.shape == (480, 640, 3)
    assert type(image) == ndarray
    # ~ uncomment to see plot if needed
    # from matplotlib.pyplot import imshow, show
    # imshow(image)
    # show()

def test_mj_state():
    # ~ test whether the mujoco state access works
    obs, info = env.reset()
    assert type(env.unwrapped.sim_state) == MujocoState


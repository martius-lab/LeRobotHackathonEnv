from lerobothackathonenv.lerobot_types import *
import numpy as np
from dataclasses import dataclass

# ~ Make a class that represents the state...
@dataclass
class MujocoState:
    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    xpos: np.ndarray
    xquat: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    sim_metadata: dict[str, Any]


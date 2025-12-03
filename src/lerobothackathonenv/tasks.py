from abc import abstractmethod
from .types import *

from numpy import clip, array, exp
from numpy.linalg import norm

import mujoco
from gymnasium import spaces

class ExtendedTask(Task, ABC):
    XML_PATH: Path
    ACTION_SPACE: Space
    OBSERVATION_SPACE: Space

    @abstractmethod
    def get_reward(self, physics: Physics) -> float:
        raise NotImplementedError("get_sim_metadata is not implemented!")

    @abstractmethod
    def get_observation(self, physics: Physics) -> Obs:
        raise NotImplementedError("get_sim_metadata is not implemented!")

    @abstractmethod
    def get_sim_metadata(self,) -> Dict:
        raise NotImplementedError("get_sim_metadata is not implemented!")


class ExampleTask(ExtendedTask):
    XML_PATH = (
            Path(__file__).parent
                / "models"
                / "xml"
                / "so101_tabletop_manipulation_generated.xml"
        )
    ACTION_SPACE = spaces.Box(low=-1, high=1, shape=(6,), dtype=float64)

    RANGE_QPOS = (-3.0, 3.0)
    RANGE_QVEL = (-10.0, 10.0)
    RANGE_AF = (-3.35, 3.35)
    RANGE_GRIPPER = (-1.5, 1.5)
    OBSERVATION_SPACE = spaces.Dict(
        dict(
            qpos=spaces.Box(*RANGE_QPOS, shape=(27,), dtype=float64),
            qvel=spaces.Box(*RANGE_QVEL, shape=(24,), dtype=float64),
            actuator_force=spaces.Box(*RANGE_AF, shape=(6,), dtype=float64),
            gripper_pos=spaces.Box(*RANGE_GRIPPER, shape=(3,), dtype=float64),
        )
    )

    def __init__(self, random=None):
        super().__init__(random=random)

    def get_observation(
        self,
        physics: Physics
    ) -> Obs:
        data = physics.data
        gripper_site_id = mujoco.mj_name2id(
            physics.model._model,
            mujoco.mjtObj.mjOBJ_SITE.value,
            "gripperframe"
        )
        gripper_pos = data.site_xpos[gripper_site_id]
        obs = dict(
            qpos=clip(data.qpos, *self.RANGE_QPOS),
            qvel=clip(data.qvel, *self.RANGE_QVEL),
            actuator_force=clip(data.actuator_force, *self.RANGE_AF),
            gripper_pos=clip(gripper_pos, *self.RANGE_GRIPPER)
        )
        return obs

class ExampleReachTask(ExampleTask):
    def __init__(self, random=None, target_pos: NDArray = array([0, 0, 1.1])):
        super(ExampleTask, self).__init__(random=random)
        self.target_pos = target_pos

    def get_reward(
        self,
        physics: Physics
    ) -> float:
        obs = self.get_observation(physics)
        gripper_pos = obs.get("gripper_pos")
        delta = self.target_pos - gripper_pos
        sigma = 0.1
        reward = exp(-norm(delta) ** 2 / ( 2 * sigma ))
        return reward

    def get_sim_metadata(self,):
        return {"target_pos": self.target_pos.copy()}

# Define more tasks here...

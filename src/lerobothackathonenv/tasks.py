from abc import ABC, abstractmethod
from lerobothackathonenv.lerobot_types import *

from numpy import clip, array, exp
from numpy.linalg import norm
import numpy as np

import mujoco
from gymnasium import spaces

class ExtendedTask(Task, ABC):
    """
    This class represents one "variation" of the
    LeRobot environment. Subclass this to create
    new variations.
    """
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

    def get_success(self, physics: Physics) -> bool:
        """
        Optional task-specific success metric.
        Defaults to False; override in concrete tasks that
        have a meaningful notion of success.
        """
        return False

# ~ Define a task without reward yet that uses the
#   a generated mujoco xml file for the physics

class ExampleTask(ExtendedTask):
    XML_PATH = (
            Path(__file__).parent
                / "models"
                / "xml"
                / "so101_convex_decomb.xml"
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

# ~ Take the task above and make it a reaching task
#   by defining the appropriate reward function

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

class GoalConditionedObjectPlaceTask(ExampleTask):
    FIXED_SEED = 42
    DELTA = 0.1
    TABLE_HEIGHT = 0.6
    RANGE_TARGET_POS = (-0.4, 0.4)
    # Rectangular workspace in front of the robot where
    # objects and goals are spawned (Meta-World style).
    OBJECT_X_RANGE = (0.07, 0.33)
    OBJECT_Y_RANGE = (-0.17, 0.17)
    GOAL_X_RANGE = (0.07, 0.33)
    GOAL_Y_RANGE = (-0.17, 0.17)
    # Reward shaping parameters
    APPROACH_SIGMA = 0.07
    PLACE_SIGMA = 0.05
    SUCCESS_TOL = 0.03
    LIFT_HEIGHT = TABLE_HEIGHT + 0.03
    REACH_WEIGHT = 0.3
    PLACE_WEIGHT = 0.7
    LIFT_BONUS = 0.2
    SUCCESS_BONUS = 1.0
    DISTRACTOR_PENALTY_WEIGHT = 1.0
    # ~ Body names of objects to be manipulated
    MANIPULATABLE_BODY_NAMES: List[str] = [
        "milk_0",
        "bread_1",
        "cereal_2",
    ]
    # Note: target_pos has x,y in RANGE_TARGET_POS and fixed z = TABLE_HEIGHT + DELTA.
    _TARGET_LOW = array([RANGE_TARGET_POS[0], RANGE_TARGET_POS[0], TABLE_HEIGHT + DELTA - 0.1])
    _TARGET_HIGH = array([RANGE_TARGET_POS[1], RANGE_TARGET_POS[1], TABLE_HEIGHT + DELTA - 0.1])
    OBSERVATION_SPACE = spaces.Dict(
        dict(
            qpos=spaces.Box(*ExampleTask.RANGE_QPOS, shape=(27,), dtype=float64),
            qvel=spaces.Box(*ExampleTask.RANGE_QVEL, shape=(24,), dtype=float64),
            actuator_force=spaces.Box(*ExampleTask.RANGE_AF, shape=(6,), dtype=float64),
            gripper_pos=spaces.Box(*ExampleTask.RANGE_GRIPPER, shape=(3,), dtype=float64),
            target_pos=spaces.Box(_TARGET_LOW, _TARGET_HIGH, shape=(3,), dtype=float64),
            object_index=spaces.Box(0, 1, shape=(len(MANIPULATABLE_BODY_NAMES), ), dtype=float64)
        )
    )


    def __init__(self, random=None):
        super(ExampleTask, self).__init__(random=random)
        # Set fixed random seed
        if self.FIXED_SEED:
            self._random = np.random.RandomState(self.FIXED_SEED)

    def _set_body_pos(
        self,
        physics: Physics,
        body_name: str,
        pos: NDArray[float64]
    ):
        """
        Helper to set the position of a body with a freejoint while
        preserving its current orientation.
        """
        model = physics.model._model
        data = physics.data

        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY.value,
            body_name
        )
        if body_id < 0:
            return

        j_addr = model.body_jntadr[body_id]
        if j_addr < 0:
            return

        qpos_adr = model.jnt_qposadr[j_addr]
        # For a free joint, qpos layout is [x, y, z, qw, qx, qy, qz]
        current_quat = data.qpos[qpos_adr + 3 : qpos_adr + 7].copy()
        data.qpos[qpos_adr : qpos_adr + 3] = pos
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = current_quat

    def initialize_episode(
        self,
        physics: Physics,
        random=None
    ):
        """
        Called by dm_control at the beginning of each episode.
        We (1) resample the goal, (2) reposition the goal marker mocap
        body, and (3) respawn objects in a reachable region.
        """
        # Set fixed random seed
        if self.FIXED_SEED:
            self._random = np.random.RandomState(self.FIXED_SEED)

        super().initialize_episode(physics)

        # New goal for this episode
        self.resample_goal()

        model = physics.model._model
        data = physics.data

        # Cache body ids for manipulatable objects once per env.
        self._manipulatable_body_ids = [
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY.value,
                name
            )
            for name in self.MANIPULATABLE_BODY_NAMES
        ]

        # Move goal_marker mocap body to match target_pos for visualization
        goal_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY.value,
            "goal_marker"
        )
        if goal_body_id >= 0:
            mocap_id = model.body_mocapid[goal_body_id]
            if mocap_id >= 0:
                data.mocap_pos[mocap_id] = self.target_pos.copy()

        # Use task-specific RNG so seeding works as expected.
        # Respawn objects uniformly in a rectangular patch
        # in front of the robot on the table and remember
        # their initial positions for distraction penalty.
        x_low, x_high = self.OBJECT_X_RANGE
        y_low, y_high = self.OBJECT_Y_RANGE
        self.initial_object_positions = []
        rng = self._random
        for body_name in self.MANIPULATABLE_BODY_NAMES:

            x = rng.uniform(x_low, x_high)
            y = rng.uniform(y_low, y_high)
            z = self.TABLE_HEIGHT + self.DELTA
            pos = array([x, y, z])
            self._set_body_pos(physics, body_name, pos)
            self.initial_object_positions.append(pos)

        # Ensure the focus object is not spawned trivially at the goal.
        focus_idx = self.focus_object
        focus_body_name = self.MANIPULATABLE_BODY_NAMES[focus_idx]
        current_pos = self.initial_object_positions[focus_idx]
        # If the focus object starts within SUCCESS_TOL of the goal,
        # resample its spawn position until it is outside this radius.
        max_tries = 20
        tries = 0
        while norm(current_pos - self.target_pos) <= self.SUCCESS_TOL and tries < max_tries:
            x = rng.uniform(x_low, x_high)
            y = rng.uniform(y_low, y_high)
            current_pos = array([x, y, z])
            tries += 1
        # Update focus object position (either original or resampled).
        self._set_body_pos(physics, focus_body_name, current_pos)
        self.initial_object_positions[focus_idx] = current_pos


    def resample_goal(self):
        # Use task-specific RNG so seeding works as expected.
        rng = self._random

        # Sample goal uniformly in the same rectangular
        # reachable patch in front of the robot.
        x_low, x_high = self.GOAL_X_RANGE
        y_low, y_high = self.GOAL_Y_RANGE

        x = rng.uniform(x_low, x_high)
        y = rng.uniform(y_low, y_high)

        z = self.TABLE_HEIGHT + self.DELTA - 0.1
        self.target_pos = array([x, y, z])
        self.focus_object = 1

    def get_reward(
        self,
        physics: Physics
    ) -> float:
        data = physics.data
        body_ids: List[int] = self._manipulatable_body_ids  # type: ignore[attr-defined]

        focus_body_id = body_ids[self.focus_object]
        object_pos = data.xpos[focus_body_id]
        gripper_site_id = mujoco.mj_name2id(
            physics.model._model,
            mujoco.mjtObj.mjOBJ_SITE.value,
            "gripperframe"
        )
        gripper_pos = data.site_xpos[gripper_site_id]


        d_obj_goal = norm(object_pos - self.target_pos)
        r = -d_obj_goal
        d_gripper_object = norm(object_pos - gripper_pos)
        r -= d_gripper_object

        return float(r)

    def get_success(self, physics: Physics) -> bool:
        """
        A rollout step is considered successful if the
        focused object is within SUCCESS_TOL of the goal.
        """
        data = physics.data
        body_ids: List[int] = self._manipulatable_body_ids  # type: ignore[attr-defined]
        focus_body_id = body_ids[self.focus_object]
        object_pos = data.xpos[focus_body_id]
        d_obj_goal = norm(object_pos - self.target_pos)
        return bool(d_obj_goal < self.SUCCESS_TOL)

    @staticmethod
    def one_hot(index: int, size: int) -> NDArray[float64]:
        vector = np.zeros(size)
        vector[index] = 1.0
        return vector

    def get_sim_metadata(self,):
        return {"target_pos": self.target_pos.copy()}

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
            qpos=clip(data.qpos, *self.RANGE_QPOS).copy(),
            qvel=clip(data.qvel, *self.RANGE_QVEL).copy(),
            actuator_force=clip(data.actuator_force, *self.RANGE_AF).copy(),
            gripper_pos=clip(gripper_pos, *self.RANGE_GRIPPER).copy(),
            target_pos=self.target_pos.copy(),
            object_index=self.one_hot(
                self.focus_object,
                len(self.MANIPULATABLE_BODY_NAMES)
            )
        )
        return obs

# Define more tasks here...

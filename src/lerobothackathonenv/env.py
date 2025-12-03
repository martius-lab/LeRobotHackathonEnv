from lerobothackathonenv.lerobot_types import *

from pathlib import Path

from .tasks import ExtendedTask, ExampleReachTask
from .structs import MujocoState

from gymnasium import Env
from dm_control import mujoco
from dm_control.rl import control
from mujoco import viewer

class LeRobot(Env):
    def __init__(
        self,
        dm_control_task_desc: Optional[ExtendedTask] = None,
    ):
        # ~ Make inner dm_control env
        self.dm_control_task: ExtendedTask = (
            dm_control_task_desc or ExampleReachTask()
        )
        dm_control_physics = mujoco.Physics.from_xml_path(
            str(self.dm_control_task.XML_PATH)
        )
        self.dm_control_env = control.Environment(
            dm_control_physics,
            self.dm_control_task
        )

        # ~ Init gym-required space variables
        self.observation_space = self.dm_control_task.OBSERVATION_SPACE
        self.action_space = self.dm_control_task.ACTION_SPACE

        # ~ Neded for the mujoco viewer
        self._window = None

    def step(
        self,
        action: NDArray[float64]
    ) -> StepResult:
        """
        Standard gym-rquired function for stepping the env
        """
        step, reward, discount, observation = self.dm_control_env.step(action)
        info: Dict = dict()
        terminated = trunctuated = False
        return observation, reward, terminated, trunctuated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult:
        """
        Standard gym-rquired function for reseting the env
        """
        super().reset(seed=seed)
        time_step = self.dm_control_env.reset()
        dummy_info: Dict = dict()
        return time_step.observation, dummy_info

    def render_to_window(self):
        """
        Renders to the mujoco viewer
        """
        if self._window is None:
            self._window = viewer.launch_passive(
                self.dm_control_env._physics.model.ptr,
                self.dm_control_env._physics.data.ptr
            )
        else:
            self._window.sync()

    def render(self,
        width=320,
        height=240,
        camera_id=-1,
    ):
        """
        Renders to a numpy array
        """
        return self.dm_control_env.physics.render(
            width=width,
            height=height,
            camera_id=camera_id
        )

    @property
    def sim_state(self) -> MujocoState:
        """
        This property is used for standardized
        representation of states in a rollout
        for later rendering and postprocessing
        """
        return MujocoState(
            time=self.dm_control_env._physics.data.time,
            qpos=self.dm_control_env._physics.data.qpos,
            qvel=self.dm_control_env._physics.data.qvel,
            xpos=self.dm_control_env._physics.data.xpos,
            xquat=self.dm_control_env._physics.data.xquat,
            mocap_pos=self.dm_control_env._physics.data.mocap_pos,
            mocap_quat=self.dm_control_env._physics.data.mocap_quat,
            sim_metadata=self.dm_control_task.get_sim_metadata(),
        )


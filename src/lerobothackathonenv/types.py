from abc import ABC
from pathlib import Path

from numpy.typing import NDArray
from numpy import float64

from typing import Dict, Tuple, Any, Optional, TypeAlias
Obs: TypeAlias = Dict[str, NDArray[float64]]
Info: TypeAlias = Dict[str, Any]
StepResult: TypeAlias = Tuple[Obs, float64, bool, bool, Info]
ResetResult: TypeAlias = Tuple[Obs, Info]

from gymnasium.spaces import Space

from dm_control.mujoco.engine import Physics
from dm_control.suite.base import Task
from .tasks import ExtendedTask

"""
Random classes used throughout the repository
to perform type annotations
"""
# ~ Paths
from pathlib import Path

# ~ Numpy
from numpy.typing import NDArray
from numpy import float64

# ~ Composite types and commonly used types
from typing import Dict, Tuple, Any, Optional, TypeAlias, List
Obs: TypeAlias = Dict[str, NDArray[float64]]
Info: TypeAlias = Dict[str, Any]
StepResult: TypeAlias = Tuple[Obs, float64, bool, bool, Info]
ResetResult: TypeAlias = Tuple[Obs, Info]

# ~ Env related
from gymnasium.spaces import Space
from dm_control.mujoco.engine import Physics
from dm_control.suite.base import Task

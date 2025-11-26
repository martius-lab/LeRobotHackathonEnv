# LeRobotHackathonEnv
Minimal, extendable LeRobot gym environment.

## Installation

```bash
# Clone the repository
git clone https://github.com/uakel/LeRobotHackathonEnv.git
cd LeRobotHackathonEnv
uv sync
```

## Verify that installation works
Run `uv run [test/mj_viewer_rendering.py](test/mj_viewer_rendering.py)` and `uv run pytest`.

## Getting started
See [`tests/`](tests/)

## StdTask
`ExtendedTask`s in the [`tasks.py`](src/lerobothackathonenv/tasks.py) script define one particular instance of the environment, that is one particulatr way to implement environment observations and reward function. A template implementation "`StdTask`" is provided.

### Observation space
The environment provides observations as a dictionary containing:
- `qpos`: Joint positions (6-dimensional, range: -3 to 3)
- `qvel`: Joint velocities (6-dimensional, range: -10 to 10)
- `actuator_force`: Actuator forces (6-dimensional, range: -3.35 to 3.35)

### Action space
- **Type**: Continuous
- **Shape**: (6,) - 6 joint torques
- **Range**: [-1, 1]

## Modifying the environment
Subclass `ExtendedTask` to modify the observations and reward:

```python
from lerobothackathonenv.types import ExtendedTask
from gymnasium import spaces

class MyCustomTask(ExtendedTask):
    ACTION_SPACE = spaces.Box(...)
    OBSERVATION_SPACE = spaces.Dict({...})

    def get_observation(self, physics):
        # Define custom observation logic
        pass

    def get_reward(self, physics):
        # Define custom reward function
        pass

# Use custom task
env = LeRobot(dm_control_task_desc=MyCustomTask())
```

We should consider to add an interface to `ExtendedTask` that standardizes adding objects and new textures to the environment.

### Writing and running tests
Add test to the test folder in functions with "test" in the name and run

```bash
# Run all tests
uv run pytest
```

## Robot Model
The robot model used is the SO-101. The motor presets are for the STS3215 servo motor model.

## Nano banana 3 style transfer test
**Prompt:** *Please generate this robot in a realistic scene for sim to real transfer*

**Result:**
![style transfer](assets/nano_banana_transfer.png)

On the top are the input and output images and on the bottom I overlayed both images with 50% opacity. One can see that the geometry is preserved pretty much perfectly.

## Acknowledgments
The xml files are taken from: https://github.com/TheRobotStudio/SO-ARM100.git

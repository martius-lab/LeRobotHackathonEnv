from gymnasium.envs.registration import register
from .tasks import GoalConditionedObjectPlaceTask

# ~ This registers the ExampleReachTask
register(
    id="LeRobot-v0",
    entry_point="lerobothackathonenv.env:LeRobot",
)

# ~ Goal conditioned task
register(
    id="LeRobotGoalConditioned-v0",
    entry_point="lerobothackathonenv.env:LeRobot",
    kwargs=dict(
        dm_control_task_desc=GoalConditionedObjectPlaceTask()
    )
)

from gymnasium.envs.registration import register

# ~ This registers the ExampleReachTask
register(
    id="LeRobot-v0",
    entry_point="lerobothackathonenv.env:LeRobot",
)

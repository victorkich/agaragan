from gym.envs.registration import register

register(
    id='TurtleBot3-v0',
    entry_point='gym_turtlebot3.envs:TurtleBot3Env'
)

max_env_size = 3

register(
    id='TurtleBot3_Circuit_Simple-v0',
    entry_point='gym_turtlebot3.envs:TurtleBot3Env',
    kwargs={'max_env_size': max_env_size}
)

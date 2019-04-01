from gym.envs.registration import register

register(
    id='firecommander-v0',
    entry_point='gym_firecommander.envs:FireCommanderEnv',
)
register(
    id='firecommander-hard-v0',
    entry_point='gym_firecommander.envs:FireCommanderHardEnv',
)

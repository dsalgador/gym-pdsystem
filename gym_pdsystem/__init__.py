import logging
from gym.envs.registration import register


logger = logging.getLogger(__name__)

register(
    id='PDSystem-v0',
    entry_point='gym_pdsystem.envs:PDSystemEnv'#,
    #timestep_limit=1000,
    #reward_threshold=1.0,
    #nondeterministic = True,
)

register(
    id='Pendulum-v0',
    entry_point='gym.envs.classic_control:PendulumEnv',
    max_episode_steps=200,
)
# register env iot be detected by Gym
from gym.envs.registration import register

register(
    id="MovingTarget-v0",
    entry_point="nmnet_environments.envs:MovingTargetEnv",
    max_episode_steps=300,
)

register(
    id="MultipleReferences-v0",
    entry_point="nmnet_environments.envs:MultipleReferencesEnv",
    max_episode_steps=300,
)

register(
    id="WindyReference-v0",
    entry_point="nmnet_environments.envs:WindyReferenceEnv",
    max_episode_steps=300,
)

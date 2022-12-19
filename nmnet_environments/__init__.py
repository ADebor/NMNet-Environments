# register env iot be detected by Gym
from gym.envs.registration import register

register(
    id="Benchmark1-v0",
    entry_point="environments.envs:Benchmark_1",
    max_episode_steps=300,
)

register(
    id="Benchmark2-v0",
    entry_point="environments.envs:Benchmark_2",
    max_episode_steps=300,
)

register(
    id="Benchmark3-v0",
    entry_point="environments.envs:Benchmark_3",
    max_episode_steps=300,
)

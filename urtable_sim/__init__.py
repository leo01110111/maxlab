"""Bimanual UR7e MuJoCo tabletop sim, packaged as a Gymnasium environment.

Installing this package registers the env id below as an import side effect,
so any Gymnasium-based RL/eval code can do:

    import gymnasium as gym
    import ur_sim  # noqa: F401 -- registers "SimBimanualUR7e-v0"

    env = gym.make("SimBimanualUR7e-v0")

See env.py:SimBimanualUR7eEnv for the observation/action spec and task.
"""

import gymnasium as gym

from .env import SimBimanualUR7eEnv

ENV_ID = "SimBimanualUR7e-v0"

if ENV_ID not in gym.envs.registry:
    gym.register(
        id=ENV_ID,
        entry_point="ur_sim.env:SimBimanualUR7eEnv",
        # No max_episode_steps here: the env truncates itself
        # (max_episode_steps kwarg on SimBimanualUR7eEnv), so wrapping it in
        # gym's own TimeLimit too would double up truncation.
    )

__all__ = ["SimBimanualUR7eEnv", "ENV_ID"]

"""One shared definition of the Phase 0 environment."""

from typing import Any, Literal

import gymnasium as gym

ENV_ID = "CarRacing-v3"
RenderMode = Literal["human", "rgb_array"] | None


def make_env(*, render_mode: RenderMode = None, max_episode_steps: int = 1000) -> gym.Env:
    """Use raw RGB observations, discrete actions and the original reward."""
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    return gym.make(
        ENV_ID,
        continuous=False,
        domain_randomize=False,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )


def reset_seeded(env: gym.Env, seed: int) -> tuple[Any, dict]:
    """The track RNG and the action sampler have independent random states."""
    env.action_space.seed(seed)
    return env.reset(seed=seed)

"""Small real Box2D runs: these tests never train a model."""

import json
import os
import subprocess
import sys

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import pytest

from neural_racing_lab.envs import make_env, reset_seeded
from neural_racing_lab.manual_control import keyboard_action
from neural_racing_lab.random_agent import run_episode


def test_observation_action_reward_and_time_limit():
    env = make_env(render_mode="rgb_array", max_episode_steps=5)
    try:
        observation, _ = reset_seeded(env, 42)
        assert observation.shape == (96, 96, 3)
        assert observation.dtype == np.uint8
        assert env.observation_space.contains(observation)
        assert env.action_space.n == 5
        for step in range(5):
            action = env.action_space.sample()
            assert env.action_space.contains(action)
            observation, reward, terminated, truncated, _ = env.step(action)
            assert env.observation_space.contains(observation)
            assert np.isfinite(reward)
            assert not terminated
            assert truncated == (step == 4)
        frame = env.render()
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert frame.dtype == np.uint8
    finally:
        env.close()


def test_seed_reproduces_track_actions_and_rewards():
    env = make_env(max_episode_steps=12)
    try:
        trajectories = []
        for _ in range(2):
            observation, _ = reset_seeded(env, 123)
            frames = [observation.copy()]
            transitions = []
            for _ in range(12):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, _ = env.step(action)
                frames.append(observation.copy())
                transitions.append((int(action), float(reward), terminated, truncated))
                if terminated or truncated:
                    break
            trajectories.append((frames, transitions))
        for first, second in zip(trajectories[0][0], trajectories[1][0], strict=True):
            np.testing.assert_array_equal(first, second)
        assert trajectories[0][1] == trajectories[1][1]
        reset_seeded(env, 124)
        different_track = env.unwrapped.track.copy()
        reset_seeded(env, 123)
        assert different_track != env.unwrapped.track
    finally:
        env.close()


def test_random_agent_returns_full_time_limited_episode():
    env = make_env(max_episode_steps=7)
    try:
        result = run_episode(env, seed=42)
        assert result.length == 7
        assert result.truncated and not result.terminated
        reset_seeded(env, 42)
        expected_return = 0.0
        for _ in range(7):
            _, reward, _, _, _ = env.step(env.action_space.sample())
            expected_return += float(reward)
        assert result.episode_return == pytest.approx(expected_return, abs=1e-9)
    finally:
        env.close()


def test_outside_playfield_is_termination():
    env = make_env(max_episode_steps=10)
    try:
        reset_seeded(env, 42)
        env.unwrapped.car.hull.position = (10000, 10000)
        _, reward, terminated, truncated, _ = env.step(0)
        assert terminated and not truncated
        assert reward == pytest.approx(-100.0)
    finally:
        env.close()


def test_random_agent_stops_at_true_terminal():
    class OutsidePlayfield(gym.Wrapper):
        def reset(self, **kwargs):
            result = self.env.reset(**kwargs)
            self.unwrapped.car.hull.position = (10000, 10000)
            return result

    env = OutsidePlayfield(make_env(max_episode_steps=10))
    try:
        result = run_episode(env, seed=42)
        assert result.length == 1
        assert result.terminated and not result.truncated
        assert result.episode_return == pytest.approx(-100.0)
    finally:
        env.close()


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_time_limit(limit):
    with pytest.raises(ValueError, match="positive"):
        make_env(max_episode_steps=limit)


@pytest.mark.parametrize(
    ("left", "right", "gas", "brake", "expected"),
    [
        (False, False, False, False, 0),
        (True, False, False, False, 2),
        (False, True, False, False, 1),
        (False, False, True, False, 3),
        (True, False, True, True, 4),
        (True, False, True, False, 2),
        (True, True, True, False, 3),
    ],
)
def test_keyboard_actions(left, right, gas, brake, expected):
    assert keyboard_action(left=left, right=right, gas=gas, brake=brake) == expected


def test_cli_metrics_and_decodable_video(tmp_path):
    output = tmp_path / "run"
    command = [
        sys.executable,
        "-m",
        "neural_racing_lab.random_agent",
        "--episodes",
        "2",
        "--max-episode-steps",
        "4",
        "--video",
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert [item["seed"] for item in report["episodes"]] == [42, 43]
    assert all(item["length"] == 4 and item["truncated"] for item in report["episodes"])
    returns = [item["episode_return"] for item in report["episodes"]]
    assert report["mean_return"] == pytest.approx(np.mean(returns))
    assert report["sample_std_return"] == pytest.approx(np.std(returns, ddof=1))
    with imageio.get_reader(output / "episode-0.mp4") as reader:
        assert reader.count_frames() == 5  # reset frame + four transitions
        assert reader.get_data(0).shape[2] == 3
    # Protect previous experiment artifacts from an accidental repeated command.
    repeated = subprocess.run(command, capture_output=True, timeout=60)
    assert repeated.returncode != 0


def test_manual_loop_headless():
    environment = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
    subprocess.run(
        [sys.executable, "-m", "neural_racing_lab.manual_control", "--frames", "3"],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

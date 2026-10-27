"""Run a uniform random policy; save full episode metrics and optional MP4."""

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

from neural_racing_lab.envs import ENV_ID, make_env, reset_seeded


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    episode_return: float
    length: int
    terminated: bool
    truncated: bool


def run_episode(env: gym.Env, *, seed: int, writer: Any = None) -> EpisodeResult:
    """Continue until an environment boundary, including TimeLimit truncation."""
    reset_seeded(env, seed)
    if writer is not None:
        writer.append_data(env.render())
    episode_return = 0.0
    length = 0
    while True:
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        episode_return += float(reward)
        length += 1
        if writer is not None:
            writer.append_data(env.render())
        if terminated or truncated:
            return EpisodeResult(seed, episode_return, length, bool(terminated), bool(truncated))


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=positive_int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episode-steps", type=positive_int, default=1000)
    parser.add_argument("--video", action="store_true", help="Record only the first episode")
    parser.add_argument("--human", action="store_true", help="Show the live random agent")
    parser.add_argument("--output", type=Path, help="New output directory; never overwritten")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.human and args.video:
        parser.error("--human and --video are separate render modes; choose one")

    output = args.output or Path("artifacts") / datetime.now(UTC).strftime(
        "random-%Y%m%dT%H%M%S%fZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    render_mode = "human" if args.human else "rgb_array" if args.video else None
    env = make_env(render_mode=render_mode, max_episode_steps=args.max_episode_steps)
    results = []
    start = time.perf_counter()
    try:
        for index in range(args.episodes):
            if args.video and index == 0:
                with imageio.get_writer(
                    output / "episode-0.mp4",
                    fps=env.metadata["render_fps"],
                    codec="libx264",
                    macro_block_size=1,
                ) as writer:
                    result = run_episode(env, seed=args.seed + index, writer=writer)
            else:
                result = run_episode(env, seed=args.seed + index)
            results.append(result)
            print(json.dumps(asdict(result)), flush=True)
    finally:
        env.close()
    elapsed = time.perf_counter() - start
    returns = np.array([item.episode_return for item in results])
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "policy": "uniform_random",
        "python": sys.version,
        "platform": platform.platform(),
        "environment": ENV_ID,
        "continuous": False,
        "domain_randomize": False,
        "max_episode_steps": args.max_episode_steps,
        "render_mode": render_mode,
        "video": "episode-0.mp4" if args.video else None,
        "versions": {name: version(name) for name in ("gymnasium", "numpy", "Box2D")},
        "episodes": [asdict(item) for item in results],
        "mean_return": float(returns.mean()),
        "sample_std_return": float(returns.std(ddof=1)) if len(results) > 1 else None,
        "elapsed_seconds": elapsed,
        "steps_per_second_including_reset_and_render": sum(r.length for r in results) / elapsed,
        "interpretation": "Infrastructure baseline, not evidence of learning or a benchmark.",
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()

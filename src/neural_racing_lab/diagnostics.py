"""Report the actual interpreter, package versions, environment and CUDA check."""

import argparse
import importlib.util
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from neural_racing_lab.envs import make_env, reset_seeded


def collect_diagnostics() -> dict:
    packages = {}
    for name in ("gymnasium", "Box2D", "pygame-ce", "numpy", "imageio", "torch", "pytest", "ruff"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }
    env = make_env(render_mode="rgb_array", max_episode_steps=5)
    try:
        observation, _ = reset_seeded(env, 42)
        assert env.observation_space.contains(observation)
        for _ in range(5):
            observation, reward, terminated, truncated, _ = env.step(0)
            assert env.observation_space.contains(observation)
            assert np.isfinite(reward)
            if terminated or truncated:
                break
        frame = env.render()
        report["environment"] = {
            "observation_shape": list(observation.shape),
            "observation_dtype": str(observation.dtype),
            "action_space": str(env.action_space),
            "render_shape": list(frame.shape),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
    finally:
        env.close()
    if importlib.util.find_spec("torch") is None:
        report["cuda"] = {"checked": False, "reason": "torch extra is not installed"}
    else:
        import torch

        available = torch.cuda.is_available()
        cuda = {"checked": True, "available": available, "runtime": torch.version.cuda}
        if available:
            # Check real GPU arithmetic, not only driver enumeration.
            x = torch.arange(64, dtype=torch.float32, device="cuda").reshape(8, 8)
            actual = (x @ x.T).cpu()
            expected = x.cpu() @ x.cpu().T
            torch.testing.assert_close(actual, expected)
            properties = torch.cuda.get_device_properties(0)
            cuda.update(
                device=properties.name,
                total_memory_bytes=properties.total_memory,
                matrix_multiply_passed=True,
            )
        report["cuda"] = cuda
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(collect_diagnostics(), indent=2, ensure_ascii=False)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Inspect static or randomized reset states for the mount task."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.mount_env import MountEnv  # noqa: E402


def configure_camera(cam) -> None:
    cam.lookat[:] = [0.0, -0.18, 0.60]
    cam.distance = 2.5
    cam.azimuth = -135
    cam.elevation = -18


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--static",
        action="store_true",
        help="Show one reset without advancing physics.",
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="Disable the default reset position/velocity noise.",
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    noise_pos = 0.0 if args.no_noise else 0.01
    noise_vel = 0.0 if args.no_noise else 0.02
    env = MountEnv(
        reset_position_noise=noise_pos,
        reset_velocity_noise=noise_vel,
    )
    _, info = env.reset(seed=args.seed)
    print("mount reset:")
    for key, value in info.items():
        print(f"  {key}: {value:.6f}")
    print("Close the viewer window to exit.")

    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            configure_camera(viewer.cam)
            zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
            episode_done = False
            while viewer.is_running():
                start = time.time()
                if not args.static and not episode_done:
                    _, _, terminated, truncated, step_info = env.step(zero_action)
                    episode_done = terminated or truncated
                    if episode_done:
                        print(
                            "episode ended: "
                            f"reason={step_info['termination_reason']} "
                            f"time={step_info['episode_seconds']:.3f}s"
                        )
                viewer.sync()
                target_dt = env.control_dt if not args.static else 1.0 / 60.0
                remaining = target_dt - (time.time() - start)
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        env.close()


if __name__ == "__main__":
    main()

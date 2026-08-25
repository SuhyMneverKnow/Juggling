"""Play the trained balance-board standing policy in the MuJoCo viewer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.balance_env import BalanceBoardStandEnv  # noqa: E402
from train.train_balance import ActorCritic  # noqa: E402


def normalize(obs: np.ndarray, state: dict) -> np.ndarray:
    mean = state["mean"]
    var = state["var"]
    return np.clip((obs - mean) / np.sqrt(var + 1e-8), -10.0, 10.0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "train" / "runs" / "balance" / "latest.pt"))
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    env = BalanceBoardStandEnv(render_mode="human", reset_noise=0.0)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    expected_obs_dim = env.observation_space.shape[0]
    expected_act_dim = env.action_space.shape[0]
    if ckpt["obs_dim"] != expected_obs_dim or ckpt["act_dim"] != expected_act_dim:
        raise ValueError(
            "Checkpoint shape does not match the current leg-only environment. "
            f"checkpoint obs/act=({ckpt['obs_dim']}, {ckpt['act_dim']}), "
            f"env obs/act=({expected_obs_dim}, {expected_act_dim}). "
            "Please retrain with train/train_balance.py."
        )
    policy = ActorCritic(ckpt["obs_dim"], ckpt["act_dim"])
    policy.load_state_dict(ckpt["model"])
    policy.eval()

    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            obs_t = torch.as_tensor(normalize(obs, ckpt["obs_norm"])).unsqueeze(0)
            with torch.no_grad():
                dist, _ = policy.dist_value(obs_t)
                action = torch.tanh(dist.mean).squeeze(0).numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            env.render()
            time.sleep(env.control_dt)
        print(f"episode={ep + 1} success={info.get('is_success')} reward={total_reward:.1f}")

    env.close()


if __name__ == "__main__":
    main()

"""Train MountEnv with the shared small PyTorch PPO implementation."""

from __future__ import annotations

import argparse
import time
from collections import deque
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.balance_env import CONTROLLED_JOINTS  # noqa: E402
from envs.mount_env import (  # noqa: E402
    MountEnv,
    OBSERVATION_DIM,
    SUCCESS_HOLD_SECONDS,
    SUCCESS_PELVIS_Z_MIN,
    SUCCESS_TORSO_TILT_MAX,
    SUCCESS_BOARD_TILT_MAX,
    FAILURE_PELVIS_Z_MIN,
    FAILURE_TORSO_TILT_MAX,
    FAILURE_BOARD_TILT_MAX,
    FAILURE_BOARD_ROLLER_HORIZONTAL_ERROR_MAX,
    FAILURE_ROBOT_BOARD_HORIZONTAL_DISTANCE_MAX,
)
from rewards.mount_rewards import REWARD_VERSION  # noqa: E402


def init_wandb(args, num_updates: int, obs_dim: int, act_dim: int):
    if not args.use_wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is not installed in the current Python environment.") from exc

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config={
            **vars(args),
            "num_updates": num_updates,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "controlled_joints": CONTROLLED_JOINTS,
            "reward_version": REWARD_VERSION,
            "observation_version": "mount_relative_102_v1",
            "expected_obs_dim": OBSERVATION_DIM,
            "success_hold_seconds": SUCCESS_HOLD_SECONDS,
            "success_pelvis_z_min": SUCCESS_PELVIS_Z_MIN,
            "success_torso_tilt_degrees": float(np.rad2deg(SUCCESS_TORSO_TILT_MAX)),
            "success_board_tilt_degrees": float(np.rad2deg(SUCCESS_BOARD_TILT_MAX)),
            "failure_pelvis_z_min": FAILURE_PELVIS_Z_MIN,
            "failure_torso_tilt_degrees": float(np.rad2deg(FAILURE_TORSO_TILT_MAX)),
            "failure_board_tilt_degrees": float(np.rad2deg(FAILURE_BOARD_TILT_MAX)),
            "failure_board_roller_error_m": FAILURE_BOARD_ROLLER_HORIZONTAL_ERROR_MAX,
            "failure_robot_board_distance_m": FAILURE_ROBOT_BOARD_HORIZONTAL_DISTANCE_MAX,
            "normalization/observation": "running_mean_variance_clip_10",
            "normalization/reward": "discounted_return_std_no_mean_subtraction",
            "normalization/advantage": "per_rollout_batch_standardization",
        },
    )


def append_info_metrics(metric_values: dict[str, list[float]], infos: dict) -> None:
    for key, value in infos.items():
        if key.startswith("_") or key == "final_info":
            continue
        arr = np.asarray(value)
        if arr.dtype == np.dtype("O"):
            continue
        arr = arr.astype(np.float64, copy=False).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            metric_values[key].append(float(arr.mean()))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.6))

    def dist_value(self, obs: torch.Tensor):
        obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        mean = torch.nan_to_num(self.actor(obs), nan=0.0, posinf=1.0, neginf=-1.0)
        log_std = torch.nan_to_num(self.log_std, nan=-0.6, posinf=1.0, neginf=-4.0)
        log_std = torch.clamp(log_std, -4.0, 1.0)
        std = torch.exp(log_std).clamp(1e-4, 3.0).expand_as(mean)
        std = torch.nan_to_num(std, nan=0.5, posinf=3.0, neginf=1e-4).clamp(1e-4, 3.0)
        value = torch.nan_to_num(self.critic(obs).squeeze(-1), nan=0.0, posinf=1e4, neginf=-1e4)
        return Normal(mean, std, validate_args=False), value


class RunningNorm:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x).all(axis=1)]
        if x.shape[0] == 0:
            return
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total
        self.mean = new_mean
        self.var = m_2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        mean = np.nan_to_num(self.mean, nan=0.0, posinf=0.0, neginf=0.0)
        var = np.nan_to_num(self.var, nan=1.0, posinf=1.0, neginf=1.0)
        return np.clip((x - mean) / np.sqrt(var + 1e-8), -10.0, 10.0).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}


class RewardNormalizer:
    """Normalize rewards by the running std of discounted episode returns.

    The mean is deliberately not subtracted: subtracting a changing reward
    mean can alter the task objective.  This follows the usual PPO/VecNormalize
    approach and only controls the scale seen by GAE and the value function.
    """

    def __init__(self, num_envs: int, gamma: float, clip: float = 10.0):
        self.gamma = gamma
        self.clip = clip
        self.returns = np.zeros(num_envs, dtype=np.float64)
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def normalize(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        rewards = np.nan_to_num(
            np.asarray(rewards, dtype=np.float64), nan=-10.0, posinf=10.0, neginf=-10.0
        )
        self.returns = self.gamma * self.returns + rewards
        self._update(self.returns)
        normalized = rewards / np.sqrt(self.var + 1e-8)
        self.returns[np.asarray(dones, dtype=bool)] = 0.0
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def _update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        batch_mean = float(values.mean())
        batch_var = float(values.var())
        batch_count = values.size
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta * delta * self.count * batch_count / total
        self.mean = new_mean
        self.var = m_2 / total
        self.count = total

    def state_dict(self):
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
            "gamma": self.gamma,
            "clip": self.clip,
        }


def make_env(
    seed: int,
    episode_seconds: float,
    reset_position_noise: float,
    reset_velocity_noise: float,
):
    def thunk():
        env = MountEnv(
            episode_seconds=episode_seconds,
            reset_position_noise=reset_position_noise,
            reset_velocity_noise=reset_velocity_noise,
        )
        env.reset(seed=seed)
        return env

    return thunk


def train(args):
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"using device={device}")

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.seed + i,
                args.episode_seconds,
                args.reset_position_noise,
                args.reset_velocity_noise,
            )
            for i in range(args.num_envs)
        ]
    )
    obs, _ = envs.reset(seed=args.seed)
    obs_norm = RunningNorm(obs.shape[1:])
    obs_norm.update(obs)
    reward_norm = RewardNormalizer(args.num_envs, args.gamma, args.reward_clip)

    obs_dim = obs.shape[1]
    act_dim = envs.single_action_space.shape[0]
    policy = ActorCritic(obs_dim, act_dim).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    success_hist = deque(maxlen=100)
    fallen_hist = deque(maxlen=100)
    board_failure_hist = deque(maxlen=100)
    out_of_area_hist = deque(maxlen=100)
    numerical_failure_hist = deque(maxlen=100)
    timeout_hist = deque(maxlen=100)
    episode_length_hist = deque(maxlen=100)
    return_hist = deque(maxlen=100)
    ep_returns = np.zeros(args.num_envs, dtype=np.float32)
    total_steps = 0
    start = time.time()
    save_dir = PROJECT_ROOT / "train" / "runs" / "mount"
    if args.run_name:
        save_dir = save_dir / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    last_good_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}

    if args.max_iterations is not None:
        num_updates = args.max_iterations
    else:
        num_updates = args.total_timesteps // (args.num_envs * args.rollout_steps)
    total_target_steps = num_updates * args.num_envs * args.rollout_steps
    wandb_run = init_wandb(args, num_updates, obs_dim, act_dim)

    for update in range(1, num_updates + 1):
        obs_buf, act_buf, logp_buf, rew_buf, raw_rew_buf, done_buf, val_buf = [], [], [], [], [], [], []
        update_losses = []
        update_pg_losses = []
        update_v_losses = []
        update_entropies = []
        update_metric_values = defaultdict(list)

        for _ in range(args.rollout_steps):
            obs_norm.update(obs)
            obs_t = torch.as_tensor(obs_norm.normalize(obs), device=device)
            with torch.no_grad():
                dist, value = policy.dist_value(obs_t)
                action_t = torch.tanh(dist.sample())
                logp_t = dist.log_prob(torch.atanh(torch.clamp(action_t, -0.999, 0.999))).sum(-1)

            action = action_t.cpu().numpy()
            next_obs, reward, terminated, truncated, infos = envs.step(action)
            done = np.logical_or(terminated, truncated)
            append_info_metrics(update_metric_values, infos)

            raw_reward = np.nan_to_num(reward, nan=-10.0, posinf=10.0, neginf=-10.0).astype(np.float32)
            normalized_reward = reward_norm.normalize(raw_reward, done)

            obs_buf.append(obs_norm.normalize(obs))
            act_buf.append(action)
            logp_buf.append(logp_t.cpu().numpy())
            rew_buf.append(normalized_reward)
            raw_rew_buf.append(raw_reward)
            done_buf.append(done.astype(np.float32))
            val_buf.append(value.cpu().numpy())

            # Keep reporting the environment's original reward. Only GAE and
            # the value target consume the normalized reward stored above.
            ep_returns += raw_reward
            if np.any(done):
                success_values = infos.get("is_success", np.zeros(args.num_envs, dtype=bool))
                fallen_values = infos.get("robot_fallen", np.zeros(args.num_envs, dtype=bool))
                board_failure_values = infos.get("board_roller_failed", np.zeros(args.num_envs, dtype=bool))
                out_of_area_values = infos.get("out_of_area", np.zeros(args.num_envs, dtype=bool))
                numerical_failure_values = infos.get("numerical_failure", np.zeros(args.num_envs, dtype=bool))
                timeout_values = infos.get("timeout", np.zeros(args.num_envs, dtype=bool))
                episode_length_values = infos.get("episode_seconds", np.zeros(args.num_envs, dtype=np.float32))
                final_infos = infos.get("final_info", [None] * args.num_envs)
                for env_i, d in enumerate(done):
                    if d:
                        info = final_infos[env_i] if final_infos is not None else None
                        success = bool(success_values[env_i])
                        fallen = bool(fallen_values[env_i])
                        board_failure = bool(board_failure_values[env_i])
                        out_of_area = bool(out_of_area_values[env_i])
                        numerical_failure = bool(numerical_failure_values[env_i])
                        timeout = bool(timeout_values[env_i])
                        episode_length = float(episode_length_values[env_i])
                        if info is not None:
                            success = bool(info.get("is_success", success))
                            fallen = bool(info.get("robot_fallen", fallen))
                            board_failure = bool(info.get("board_roller_failed", board_failure))
                            out_of_area = bool(info.get("out_of_area", out_of_area))
                            numerical_failure = bool(info.get("numerical_failure", numerical_failure))
                            timeout = bool(info.get("timeout", timeout))
                            episode_length = float(info.get("episode_seconds", episode_length))
                        success_hist.append(float(success))
                        fallen_hist.append(float(fallen))
                        board_failure_hist.append(float(board_failure))
                        out_of_area_hist.append(float(out_of_area))
                        numerical_failure_hist.append(float(numerical_failure))
                        timeout_hist.append(float(timeout))
                        episode_length_hist.append(episode_length)
                        return_hist.append(float(ep_returns[env_i]))
                ep_returns[done] = 0.0

            obs = next_obs
            total_steps += args.num_envs

        with torch.no_grad():
            next_value = policy.dist_value(torch.as_tensor(obs_norm.normalize(obs), device=device))[1].cpu().numpy()

        rewards = np.asarray(rew_buf)
        dones = np.asarray(done_buf)
        values = np.asarray(val_buf)
        advantages = np.zeros_like(rewards)
        lastgaelam = np.zeros(args.num_envs, dtype=np.float32)
        for t in reversed(range(args.rollout_steps)):
            next_nonterminal = 1.0 - dones[t]
            next_values = next_value if t == args.rollout_steps - 1 else values[t + 1]
            delta = rewards[t] + args.gamma * next_values * next_nonterminal - values[t]
            lastgaelam = delta + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values

        b_obs = torch.as_tensor(np.asarray(obs_buf).reshape(-1, obs_dim), device=device)
        b_obs = torch.nan_to_num(b_obs, nan=0.0, posinf=10.0, neginf=-10.0)
        b_act = torch.as_tensor(np.asarray(act_buf).reshape(-1, act_dim), device=device)
        b_logp = torch.as_tensor(np.asarray(logp_buf).reshape(-1), device=device)
        b_adv = torch.as_tensor(np.nan_to_num(advantages.reshape(-1), nan=0.0), device=device)
        b_ret = torch.as_tensor(np.nan_to_num(returns.reshape(-1), nan=0.0), device=device)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        batch_size = b_obs.shape[0]
        inds = np.arange(batch_size)
        for _ in range(args.epochs):
            np.random.shuffle(inds)
            for start_idx in range(0, batch_size, args.minibatch_size):
                mb = inds[start_idx : start_idx + args.minibatch_size]
                dist, value = policy.dist_value(b_obs[mb])
                raw_act = torch.atanh(torch.clamp(b_act[mb], -0.999, 0.999))
                logp = dist.log_prob(raw_act).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                ratio = torch.exp(logp - b_logp[mb])
                pg_loss = -torch.min(ratio * b_adv[mb], torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * b_adv[mb]).mean()
                v_loss = 0.5 * torch.square(value - b_ret[mb]).mean()
                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * entropy
                if not torch.isfinite(loss):
                    continue

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    policy.log_std.clamp_(-4.0, 1.0)

                update_losses.append(float(loss.detach().cpu()))
                update_pg_losses.append(float(pg_loss.detach().cpu()))
                update_v_losses.append(float(v_loss.detach().cpu()))
                update_entropies.append(float(entropy.detach().cpu()))

                params_finite = all(torch.isfinite(p).all() for p in policy.parameters())
                if params_finite:
                    last_good_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
                else:
                    policy.load_state_dict(last_good_state)
                    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
                    print("warning: restored last finite policy state after non-finite parameter update")

        if update % args.log_interval == 0 or update == 1:
            success = np.mean(success_hist) if success_hist else 0.0
            fallen = np.mean(fallen_hist) if fallen_hist else 0.0
            board_failure = np.mean(board_failure_hist) if board_failure_hist else 0.0
            out_of_area = np.mean(out_of_area_hist) if out_of_area_hist else 0.0
            numerical_failure = np.mean(numerical_failure_hist) if numerical_failure_hist else 0.0
            timeout = np.mean(timeout_hist) if timeout_hist else 0.0
            episode_length = np.mean(episode_length_hist) if episode_length_hist else 0.0
            ret = np.mean(return_hist) if return_hist else 0.0
            fps = int(total_steps / max(time.time() - start, 1e-6))
            loss_mean = np.mean(update_losses) if update_losses else float("nan")
            pg_loss_mean = np.mean(update_pg_losses) if update_pg_losses else float("nan")
            v_loss_mean = np.mean(update_v_losses) if update_v_losses else float("nan")
            entropy_mean = np.mean(update_entropies) if update_entropies else float("nan")
            print(
                f"update={update:04d} steps={total_steps} success={success:.2f} "
                f"fallen={fallen:.2f} board_fail={board_failure:.2f} timeout={timeout:.2f} "
                f"return={ret:.1f} loss={loss_mean:.3f} pg={pg_loss_mean:.3f} "
                f"v={v_loss_mean:.3f} ent={entropy_mean:.3f} fps={fps}"
            )
            if wandb_run is not None:
                log_data = {
                    "train/update": update,
                    "train/steps": total_steps,
                    "train/target_steps": total_target_steps,
                    "train/fps": fps,
                    "train/success_rate_100": success,
                    "train/robot_fallen_rate_100": fallen,
                    "train/board_failure_rate_100": board_failure,
                    "train/out_of_area_rate_100": out_of_area,
                    "train/numerical_failure_rate_100": numerical_failure,
                    "train/timeout_rate_100": timeout,
                    "train/episode_length_seconds_100": episode_length,
                    "train/episode_return_100": ret,
                    "train/rollout_reward_mean": float(np.mean(raw_rew_buf)),
                    "train/rollout_reward_normalized_mean": float(np.mean(rewards)),
                    "train/reward_return_running_std": float(np.sqrt(reward_norm.var + 1e-8)),
                    "normalization/observation_running_count": float(obs_norm.count),
                    "normalization/observation_mean_abs": float(np.mean(np.abs(obs_norm.mean))),
                    "normalization/observation_std_mean": float(np.mean(np.sqrt(obs_norm.var + 1e-8))),
                    "loss/total": loss_mean,
                    "loss/policy": pg_loss_mean,
                    "loss/value": v_loss_mean,
                    "loss/entropy": entropy_mean,
                }
                for key, values in update_metric_values.items():
                    prefix = "reward" if "reward" in key or "penalty" in key else "metric"
                    log_data[f"{prefix}/{key}"] = float(np.mean(values))
                wandb_run.log(log_data, step=total_steps)

        if update % args.save_interval == 0 or update == num_updates:
            torch.save(
                {
                    "model": policy.state_dict(),
                    "obs_norm": obs_norm.state_dict(),
                    "reward_norm": reward_norm.state_dict(),
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                    "controlled_joints": CONTROLLED_JOINTS,
                    "task": "mount",
                    "reward_version": REWARD_VERSION,
                    "observation_version": "mount_relative_102_v1",
                    "normalization": {
                        "observation": "running_mean_variance_clip_10",
                        "reward": "discounted_return_std_no_mean_subtraction",
                        "advantage": "per_rollout_batch_standardization",
                    },
                    "episode_seconds": args.episode_seconds,
                    "max_iterations": args.max_iterations,
                },
                save_dir / "latest.pt",
            )

    envs.close()
    if wandb_run is not None:
        wandb_run.finish()
    print(f"saved {save_dir / 'latest.pt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--max-iterations", type=int, default=7000)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--reward-clip", type=float, default=10.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--episode-seconds", type=float, default=4.0)
    parser.add_argument("--reset-position-noise", type=float, default=0.01)
    parser.add_argument("--reset-velocity-noise", type=float, default=0.02)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=20)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="mount")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    train(parser.parse_args())


if __name__ == "__main__":
    main()

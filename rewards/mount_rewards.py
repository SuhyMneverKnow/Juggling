"""Reward interface for the mount task.

The reward is intentionally left at zero while the PPO pipeline is validated.
Add shaped terms in :func:`mount_reward` without changing the environment or
training-loop interface.
"""

from __future__ import annotations

import numpy as np


REWARD_VERSION = "placeholder_zero_v0"


def mount_reward(
    env,
    action: np.ndarray,
    previous_action: np.ndarray,
    task_info: dict[str, float | bool],
) -> tuple[float, dict[str, float]]:
    """Return the mount reward and named logging terms.

    This placeholder deliberately provides no learning signal. It exists so
    the environment, normalization, PPO update, checkpointing, and logging can
    be tested before reward design begins.
    """
    del env, action, previous_action, task_info
    return 0.0, {
        "placeholder_reward": 0.0,
        "total_reward": 0.0,
    }

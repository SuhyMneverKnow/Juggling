"""Reward terms for leg-only H1-2 balance-board stabilization."""

from __future__ import annotations

import numpy as np


TORSO_TILT_REFERENCE = np.deg2rad(20.0)
TORSO_LATERAL_TILT_REFERENCE = np.deg2rad(15.0)
TORSO_TILT_RATE_REFERENCE = 1.0  # rad/s
TORSO_DOWNWARD_SPEED_REFERENCE = 0.5  # m/s
TORSO_LATERAL_TILT_WEIGHT = 0.5
TORSO_TILT_RATE_WEIGHT = 0.25
TORSO_DOWNWARD_SPEED_WEIGHT = 0.25
FOOT_HORIZONTAL_ERROR_REFERENCE = 0.08  # m, stable fixed-stance radius
FOOT_HORIZONTAL_PENALTY_WEIGHT = 0.5
LEG_ACTION_RATE_WEIGHT = 0.02
ARM_ACTION_RATE_WEIGHT = 0.02
LEG_ACTION_DIMS = 10
BOARD_X_VELOCITY_REFERENCE = 0.30  # m/s
ROLLER_X_VELOCITY_REFERENCE = 0.30  # m/s
BOARD_PITCH_RATE_REFERENCE = 0.80  # rad/s
SYSTEM_CENTER_ERROR_REFERENCE = 0.40  # m


def exp_tracking(error: float, scale: float) -> float:
    return float(np.exp(-scale * error))


def alive() -> float:
    return 1.0


def torso_stability_penalty(env) -> tuple[float, dict[str, float]]:
    """Penalize tilt, outward tipping, and downward torso motion.

    The best value is zero at a vertical, stationary torso. Tilt direction is
    irrelevant because the angle is unsigned. Only positive tilt rate is
    penalized, so corrective motion back toward vertical is not discouraged.
    """
    tilt_angle = env.torso_tilt_angle()
    lateral_tilt, sagittal_tilt = env.torso_tilt_components()
    outward_tilt_rate = max(0.0, env.torso_tilt_rate())
    downward_speed = env.torso_downward_speed()

    angle_penalty = -float(np.square(tilt_angle / TORSO_TILT_REFERENCE))
    lateral_tilt_penalty = -TORSO_LATERAL_TILT_WEIGHT * float(
        np.square(lateral_tilt / TORSO_LATERAL_TILT_REFERENCE)
    )
    tilt_rate_penalty = -TORSO_TILT_RATE_WEIGHT * float(
        np.square(outward_tilt_rate / TORSO_TILT_RATE_REFERENCE)
    )
    downward_speed_penalty = -TORSO_DOWNWARD_SPEED_WEIGHT * float(
        np.square(downward_speed / TORSO_DOWNWARD_SPEED_REFERENCE)
    )
    total = float(
        np.clip(
            angle_penalty + lateral_tilt_penalty + tilt_rate_penalty + downward_speed_penalty,
            -10.0,
            0.0,
        )
    )
    return total, {
        "torso_tilt_angle_rad": tilt_angle,
        "torso_tilt_angle_deg": float(np.rad2deg(tilt_angle)),
        "torso_lateral_tilt_rad": lateral_tilt,
        "torso_lateral_tilt_deg": float(np.rad2deg(lateral_tilt)),
        "torso_sagittal_tilt_rad": sagittal_tilt,
        "torso_sagittal_tilt_deg": float(np.rad2deg(sagittal_tilt)),
        "torso_outward_tilt_rate": outward_tilt_rate,
        "torso_downward_speed": downward_speed,
        "torso_angle_penalty": angle_penalty,
        "torso_lateral_tilt_penalty": lateral_tilt_penalty,
        "torso_tilt_rate_penalty": tilt_rate_penalty,
        "torso_downward_speed_penalty": downward_speed_penalty,
    }


def upper_body_still(env) -> float:
    q_error = env.upper_body_joint_error()
    ang_vel = env.torso_angular_velocity_norm()
    return exp_tracking(q_error + 0.05 * ang_vel * ang_vel, 8.0)


def feet_relative_pose(env) -> float:
    left_error, right_error = env.feet_board_pose_errors()
    return exp_tracking(left_error + right_error, 80.0)


def feet_fixed_stance_penalty(env) -> tuple[float, dict[str, float]]:
    """Continuously penalize horizontal drift from each reset foot position."""
    left_error, right_error = env.feet_board_horizontal_errors()
    penalty = -FOOT_HORIZONTAL_PENALTY_WEIGHT * (
        np.square(left_error / FOOT_HORIZONTAL_ERROR_REFERENCE)
        + np.square(right_error / FOOT_HORIZONTAL_ERROR_REFERENCE)
    )
    return float(np.clip(penalty, -5.0, 0.0)), {
        "left_foot_horizontal_error": left_error,
        "right_foot_horizontal_error": right_error,
    }


def board_level(env) -> float:
    board_up = env.board_up()
    return float(np.exp(6.0 * (board_up - 1.0)))


def roller_center(env) -> float:
    return exp_tracking(env.board_roller_xy_error() ** 2, 20.0)


def pelvis_height(env) -> float:
    return exp_tracking((env.pelvis_height() - env.init_pelvis_height) ** 2, 30.0)


def board_motion_penalty(env) -> tuple[float, dict[str, float]]:
    """Penalize task-relevant drift without discouraging roller rotation."""
    board_vx = float(env.data.qvel[env.board_x_qvel_id])
    roller_vx = float(env.data.qvel[env.roller_x_qvel_id])
    board_pitch_rate = float(env.data.qvel[env.board_pitch_qvel_id])

    board_vx_term = float(np.square(board_vx / BOARD_X_VELOCITY_REFERENCE))
    roller_vx_term = float(np.square(roller_vx / ROLLER_X_VELOCITY_REFERENCE))
    pitch_rate_term = float(np.square(board_pitch_rate / BOARD_PITCH_RATE_REFERENCE))
    motion_penalty = float(
        np.clip(
            0.50 * board_vx_term
            + 0.20 * roller_vx_term
            + 0.30 * pitch_rate_term,
            0.0,
            5.0,
        )
    )

    system_center_x = float(
        0.5
        * (
            env.data.xpos[env.board_body, 0]
            + env.data.xpos[env.roller_body, 0]
        )
    )
    system_center_error = system_center_x - env.init_system_center_x
    system_drift_penalty = float(
        np.clip(
            np.square(system_center_error / SYSTEM_CENTER_ERROR_REFERENCE),
            0.0,
            5.0,
        )
    )
    return motion_penalty, {
        "board_x_velocity": board_vx,
        "roller_x_velocity": roller_vx,
        "board_pitch_rate": board_pitch_rate,
        "system_center_x_error": system_center_error,
        "system_drift_penalty": system_drift_penalty,
    }


def leg_joint_velocity_penalty(env) -> float:
    qvel = np.clip(env.data.qvel[env.qvel_ids], -50.0, 50.0)
    return float(np.mean(np.square(qvel)))


def action_penalty(action: np.ndarray) -> float:
    return float(np.mean(np.square(action)))


def action_rate_penalty(
    action: np.ndarray, previous_action: np.ndarray
) -> tuple[float, dict[str, float]]:
    """Penalize abrupt target changes while retaining large smooth motions."""
    delta_action = np.asarray(action) - np.asarray(previous_action)
    leg_delta = delta_action[:LEG_ACTION_DIMS]
    arm_delta = delta_action[LEG_ACTION_DIMS:]
    leg_delta_squared = float(np.mean(np.square(leg_delta)))
    arm_delta_squared = float(np.mean(np.square(arm_delta)))
    leg_penalty = -LEG_ACTION_RATE_WEIGHT * leg_delta_squared
    arm_penalty = -ARM_ACTION_RATE_WEIGHT * arm_delta_squared
    return leg_penalty + arm_penalty, {
        "leg_action_rate_penalty": leg_penalty,
        "arm_action_rate_penalty": arm_penalty,
        "leg_delta_action_rms": float(np.sqrt(leg_delta_squared)),
        "arm_delta_action_rms": float(np.sqrt(arm_delta_squared)),
    }


def leg_only_balance_reward(
    env, action: np.ndarray, previous_action: np.ndarray
) -> tuple[float, dict]:
    torso_penalty, torso_diagnostics = torso_stability_penalty(env)
    feet_penalty, feet_diagnostics = feet_fixed_stance_penalty(env)
    action_rate, action_rate_diagnostics = action_rate_penalty(action, previous_action)
    motion_penalty, motion_diagnostics = board_motion_penalty(env)
    terms = {
        "alive_reward": alive(),
        "torso_stability_penalty": torso_penalty,
        "upper_body_still_reward": upper_body_still(env),
        "feet_relative_pose_reward": feet_relative_pose(env),
        "feet_fixed_stance_penalty": feet_penalty,
        "board_level_reward": board_level(env),
        "roller_center_reward": roller_center(env),
        "pelvis_height_reward": pelvis_height(env),
        "board_motion_penalty": motion_penalty,
        "leg_joint_velocity_penalty": leg_joint_velocity_penalty(env),
        "action_penalty": action_penalty(action),
        "action_rate_penalty": action_rate,
        **torso_diagnostics,
        **feet_diagnostics,
        **action_rate_diagnostics,
        **motion_diagnostics,
    }

    # Active reward: survival, torso stability, board level, roller centering,
    # and maintaining the pelvis near its initial standing height. The torso
    # term is a non-positive penalty whose optimum is zero.
    reward = (
        1.0 * terms["alive_reward"]
        + 2.0 * terms["torso_stability_penalty"]
        + 1.0 * terms["feet_fixed_stance_penalty"]
        + 1.5 * terms["board_level_reward"]
        + 1.0 * terms["roller_center_reward"]
        + 0.5 * terms["pelvis_height_reward"]
        + 1.0 * terms["action_rate_penalty"]
        - 0.25 * terms["board_motion_penalty"]
        - 0.05 * terms["system_drift_penalty"]
    )
    if not np.isfinite(reward):
        reward = -10.0
    return float(np.clip(reward, -10.0, 10.0)), terms

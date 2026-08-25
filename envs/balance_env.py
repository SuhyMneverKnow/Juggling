"""Gymnasium environment for lateral H1-2 balance-board stabilization.

The policy outputs joint-position residuals. This environment converts those
residuals into torque commands with a PD controller. The initial pose starts
both elbows at 90 degrees. The action space controls hip/ankle roll,
hip/knee/ankle pitch, shoulder pitch/roll, and elbow joints.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from rewards.balance_rewards import leg_only_balance_reward


PROJECT_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = PROJECT_ROOT / "models" / "balance.xml"


CONTROLLED_JOINTS = [
    # Lateral leg balance.
    "left_hip_roll_joint",
    "left_ankle_roll_joint",
    "right_hip_roll_joint",
    "right_ankle_roll_joint",

    # Sagittal leg balance and knee flexion.
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",

    # Arm motion.
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_joint",
]

# Per-joint position-residual ranges in radians, in exactly the same order as
# CONTROLLED_JOINTS. These scales expose a substantially larger part of the
# joints' usable range for active leg and arm balance. The policy action itself
# remains normalized to [-1, 1].
ACTION_SCALES = np.array(
    [
        1.0,  # left hip roll
        1.0,  # left ankle roll
        1.0,  # right hip roll
        1.0,  # right ankle roll
        1.0,  # left hip pitch
        1.0,  # left knee
        1.0,  # left ankle pitch
        1.0,  # right hip pitch
        1.0,  # right knee
        1.0,  # right ankle pitch
        1.0,  # left shoulder pitch
        1.0,  # left shoulder roll
        1.0,  # left elbow
        1.0,  # right shoulder pitch
        1.0,  # right shoulder roll
        1.0,  # right elbow
    ],
    dtype=np.float64,
)

UPPER_BODY_JOINTS = [
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

INIT_JOINT_POS = {
    # Slight symmetric knee flexion avoids the straight-leg kinematic
    # singularity while preserving level feet and an upright torso.
    "left_hip_pitch_joint": -0.07,
    "left_knee_joint": 0.14,
    "left_ankle_pitch_joint": -0.07,
    "right_hip_pitch_joint": -0.07,
    "right_knee_joint": 0.14,
    "right_ankle_pitch_joint": -0.07,
    "left_elbow_joint": np.pi / 2.0,
    "right_elbow_joint": np.pi / 2.0,
}

# A completed episode is considered stable when the robot satisfies all
# posture/board conditions for at least this fraction of the evaluated steps.
# This permits brief deviations followed by active recovery.
STABLE_RATIO_THRESHOLD = 0.80

# Success thresholds describe the desired stable region. Position errors are
# expressed in metres; torso_up and board_up are dimensionless cosine values.
SUCCESS_TORSO_TILT_MAX = np.deg2rad(20.0)
SUCCESS_TORSO_UP_MIN = float(np.cos(SUCCESS_TORSO_TILT_MAX))
SUCCESS_PELVIS_Z_MIN = 0.75
SUCCESS_FOOT_BOARD_ERROR_MAX = 0.08
SUCCESS_BOARD_UP_MIN = 0.85
SUCCESS_BOARD_ROLLER_ERROR_MAX = 0.30

# Failure thresholds describe the wider admissible region outside which an
# episode terminates. Position errors are expressed in metres.
FAILURE_PELVIS_Z_MIN = 0.85
FAILURE_TORSO_TILT_MAX = np.deg2rad(45.0)
FAILURE_TORSO_UP_MIN = float(np.cos(FAILURE_TORSO_TILT_MAX))
FAILURE_BOARD_UP_MIN = 0.35
FAILURE_BOARD_ROLLER_ERROR_MAX = 0.45
FAILURE_FOOT_BOARD_ERROR_MAX = 0.16


class BalanceBoardStandEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        episode_seconds: float = 6.0,
        settling_seconds: float = 1.0,
        control_dt: float = 0.02,
        reset_noise: float = 0.0,
    ) -> None:
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None

        self.control_dt = control_dt
        self.frame_skip = max(1, int(round(control_dt / self.model.opt.timestep)))
        self.max_steps = int(round(episode_seconds / control_dt))
        self.settling_steps = int(round(settling_seconds / control_dt))
        self.action_scales = ACTION_SCALES.copy()
        self.reset_noise = reset_noise

        self.init_qpos = self.model.qpos0.copy()
        for joint_name, joint_pos in INIT_JOINT_POS.items():
            self.init_qpos[self._joint_qpos_id(joint_name)] = joint_pos
        self.init_qvel = np.zeros(self.model.nv, dtype=np.float64)

        self.pelvis_body = self._body_id("pelvis")
        self.torso_body = self._body_id("torso_link")
        self.board_body = self._body_id("board")
        self.roller_body = self._body_id("roller")
        self.left_foot_site = self._site_id("left_foot")
        self.right_foot_site = self._site_id("right_foot")

        self.actuator_ids = np.array([self._actuator_id(j) for j in CONTROLLED_JOINTS], dtype=np.int32)
        self.qpos_ids = np.array([self._joint_qpos_id(j) for j in CONTROLLED_JOINTS], dtype=np.int32)
        self.qvel_ids = np.array([self._joint_dof_id(j) for j in CONTROLLED_JOINTS], dtype=np.int32)
        self.board_x_qvel_id = self._joint_dof_id("board_x")
        self.roller_x_qvel_id = self._joint_dof_id("roller_x")
        self.board_pitch_qvel_id = self._joint_dof_id("board_pitch")
        self.default_joint_pos = self.init_qpos[self.qpos_ids].copy()
        self.upper_body_qpos_ids = np.array([self._joint_qpos_id(j) for j in UPPER_BODY_JOINTS], dtype=np.int32)
        self.upper_body_qvel_ids = np.array([self._joint_dof_id(j) for j in UPPER_BODY_JOINTS], dtype=np.int32)
        self.default_upper_body_joint_pos = self.init_qpos[self.upper_body_qpos_ids].copy()

        self.all_actuator_joint_qpos = np.array(
            [self._joint_qpos_id(self._actuator_name(i)) for i in range(self.model.nu)], dtype=np.int32
        )
        self.all_actuator_joint_dof = np.array(
            [self._joint_dof_id(self._actuator_name(i)) for i in range(self.model.nu)], dtype=np.int32
        )
        self.all_default_joint_pos = self.init_qpos[self.all_actuator_joint_qpos].copy()

        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        self.kp = np.full(self.model.nu, 45.0)
        self.kd = np.full(self.model.nu, 2.0)
        for i, name in enumerate(self._actuator_name(a) for a in range(self.model.nu)):
            if "ankle" in name:
                self.kp[i], self.kd[i] = 30.0, 1.2
            elif "shoulder" in name or "elbow" in name or "wrist" in name:
                self.kp[i], self.kd[i] = 12.0, 0.6
            elif name.startswith(("L_", "R_")):
                self.kp[i], self.kd[i] = 1.0, 0.05

        self.last_action = np.zeros(len(CONTROLLED_JOINTS), dtype=np.float32)
        mujoco.mj_forward(self.model, self.data)
        self.init_system_center_x = float(
            0.5
            * (
                self.data.xpos[self.board_body, 0]
                + self.data.xpos[self.roller_body, 0]
            )
        )
        self.init_left_foot_board = self._site_pos_in_board(self.left_foot_site)
        self.init_right_foot_board = self._site_pos_in_board(self.right_foot_site)
        self.init_pelvis_height = float(self.data.xpos[self.pelvis_body, 2])
        obs_dim = self._get_obs().shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(CONTROLLED_JOINTS),), dtype=np.float32)
        self.step_count = 0
        self.evaluated_stability_steps = 0
        self.stable_steps = 0
        self.strict_conditions_held = True

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        self.evaluated_stability_steps = 0
        self.stable_steps = 0
        self.strict_conditions_held = True
        self.last_action[:] = 0.0

        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        if self.reset_noise > 0:
            noise = self.np_random.uniform(-self.reset_noise, self.reset_noise, size=len(self.qpos_ids))
            self.data.qpos[self.qpos_ids] += noise

        mujoco.mj_forward(self.model, self.data)
        self.init_left_foot_board = self._site_pos_in_board(self.left_foot_site)
        self.init_right_foot_board = self._site_pos_in_board(self.right_foot_site)
        self._apply_pd(np.zeros_like(self.last_action))
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        previous_action = self.last_action.copy()
        for _ in range(self.frame_skip):
            self._apply_pd(action)
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        self.last_action = action.astype(np.float32)
        obs = self._get_obs()
        terminated = self._is_failed()

        # Ignore the initial settling period, then measure how often all
        # stability conditions are met. Unlike the old logical-AND criterion,
        # one transient deviation no longer makes success impossible.
        if self.step_count > self.settling_steps:
            stable_now = self._success_conditions_now()
            self.evaluated_stability_steps += 1
            self.stable_steps += int(stable_now)
            self.strict_conditions_held = self.strict_conditions_held and stable_now

        truncated = self.step_count >= self.max_steps
        survival_success = bool(truncated and not terminated)
        stable_ratio = self._stable_ratio()
        is_success = bool(survival_success and stable_ratio >= STABLE_RATIO_THRESHOLD)
        strict_success = bool(survival_success and self.strict_conditions_held)
        reward, reward_info = self._reward(action, previous_action)
        info = {
            # Primary success: survive the full episode and remain stable for
            # at least STABLE_RATIO_THRESHOLD of the post-settling interval.
            "is_success": is_success,
            # Separate survival from control quality, following the evaluation
            # style commonly used for continuous locomotion tasks.
            "survival_success": survival_success,
            "stable_ratio": stable_ratio,
            # Retain the old every-step criterion as a diagnostic only.
            "strict_success": strict_success,
            "episode_length_seconds": self.step_count * self.control_dt,
            **reward_info,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return None
        if self.viewer is None:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.lookat[:] = [0.0, 0.0, 0.75]
            self.viewer.cam.distance = 2.8
            self.viewer.cam.azimuth = -135
            self.viewer.cam.elevation = -18
        self.viewer.sync()
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _apply_pd(self, action: np.ndarray) -> None:
        target = self.all_default_joint_pos.copy()
        # The policy commands joint-position residuals; the existing PD loop
        # remains responsible for converting those targets into motor torques.
        target[self.actuator_ids] = self.default_joint_pos + self.action_scales * action
        q = self.data.qpos[self.all_actuator_joint_qpos]
        dq = self.data.qvel[self.all_actuator_joint_dof]
        torque = self.kp * (target - q) - self.kd * dq
        self.data.ctrl[:] = np.clip(torque, self.ctrl_low, self.ctrl_high)

    def _get_obs(self) -> np.ndarray:
        pelvis_quat = self.data.xquat[self.pelvis_body].copy()
        torso_quat = self.data.xquat[self.torso_body].copy()
        board_quat = self.data.xquat[self.board_body].copy()
        roller_pos = self.data.xpos[self.roller_body].copy()
        board_pos = self.data.xpos[self.board_body].copy()
        pelvis_pos = self.data.xpos[self.pelvis_body].copy()

        obs = np.concatenate(
            [
                pelvis_pos,
                pelvis_quat,
                torso_quat,
                board_pos,
                board_quat,
                roller_pos,
                self._site_pos_in_board(self.left_foot_site) - self.init_left_foot_board,
                self._site_pos_in_board(self.right_foot_site) - self.init_right_foot_board,
                self.data.cvel[self.pelvis_body],
                self.data.cvel[self.board_body],
                self.data.cvel[self.roller_body],
                self.data.qpos[self.qpos_ids] - self.default_joint_pos,
                self.data.qvel[self.qvel_ids] * 0.1,
                self.data.qpos[self.upper_body_qpos_ids] - self.default_upper_body_joint_pos,
                self.data.qvel[self.upper_body_qvel_ids] * 0.1,
                self.last_action,
            ]
        )
        return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)

    def _reward(
        self, action: np.ndarray, previous_action: np.ndarray
    ) -> tuple[float, dict]:
        reward, terms = leg_only_balance_reward(self, action, previous_action)
        terms.update(
            {
                "torso_up_value": self.torso_up(),
                "board_up_value": self.board_up(),
                "pelvis_z": self.pelvis_height(),
                "board_roller_error": self.board_roller_xy_error(),
                "left_foot_board_error": self.feet_board_pose_errors()[0],
                "right_foot_board_error": self.feet_board_pose_errors()[1],
            }
        )
        return reward, terms

    def _is_failed(self) -> bool:
        pelvis_z = self.pelvis_height()
        torso_up = self.torso_up()
        board_up = self.board_up()
        board_roller_error = self.board_roller_xy_error()
        left_foot_error, right_foot_error = self.feet_board_pose_errors()
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            return True
        if not np.isfinite([pelvis_z, torso_up, board_up, board_roller_error, left_foot_error, right_foot_error]).all():
            return True
        return bool(
            pelvis_z < FAILURE_PELVIS_Z_MIN
            or torso_up < FAILURE_TORSO_UP_MIN
            or board_up < FAILURE_BOARD_UP_MIN
            or board_roller_error > FAILURE_BOARD_ROLLER_ERROR_MAX
            or left_foot_error > FAILURE_FOOT_BOARD_ERROR_MAX
            or right_foot_error > FAILURE_FOOT_BOARD_ERROR_MAX
        )

    def _stable_ratio(self) -> float:
        if self.evaluated_stability_steps == 0:
            return 0.0
        return float(self.stable_steps / self.evaluated_stability_steps)

    def _success_conditions_now(self) -> bool:
        left_foot_error, right_foot_error = self.feet_board_pose_errors()
        return bool(
            self.torso_up() > SUCCESS_TORSO_UP_MIN
            and self.pelvis_height() > SUCCESS_PELVIS_Z_MIN
            and left_foot_error < SUCCESS_FOOT_BOARD_ERROR_MAX
            and right_foot_error < SUCCESS_FOOT_BOARD_ERROR_MAX
            and self.board_up() > SUCCESS_BOARD_UP_MIN
            and self.board_roller_xy_error() < SUCCESS_BOARD_ROLLER_ERROR_MAX
        )

    def _site_pos_in_board(self, site_id: int) -> np.ndarray:
        board_pos = self.data.xpos[self.board_body]
        board_rot = self.data.xmat[self.board_body].reshape(3, 3)
        site_pos = self.data.site_xpos[site_id]
        return board_rot.T @ (site_pos - board_pos)

    def feet_board_pose_errors(self) -> tuple[float, float]:
        """Return each foot site's displacement from reset in board coordinates.

        Values are Euclidean distances in metres.  The previous implementation
        returned squared distances (m^2), which made thresholds such as 0.08
        physically mean sqrt(0.08) metres instead of 8 cm.
        """
        left = self._site_pos_in_board(self.left_foot_site)
        right = self._site_pos_in_board(self.right_foot_site)
        return (
            float(np.linalg.norm(left - self.init_left_foot_board)),
            float(np.linalg.norm(right - self.init_right_foot_board)),
        )

    def feet_board_horizontal_errors(self) -> tuple[float, float]:
        """Return reset-relative foot-site XY distances in the board frame, in metres."""
        left = self._site_pos_in_board(self.left_foot_site) - self.init_left_foot_board
        right = self._site_pos_in_board(self.right_foot_site) - self.init_right_foot_board
        return float(np.linalg.norm(left[:2])), float(np.linalg.norm(right[:2]))

    def upper_body_joint_error(self) -> float:
        err = self.data.qpos[self.upper_body_qpos_ids] - self.default_upper_body_joint_pos
        return float(np.mean(np.square(err)))

    def torso_angular_velocity_norm(self) -> float:
        angular_velocity, _ = self.torso_world_velocity()
        return float(np.linalg.norm(np.clip(angular_velocity, -20.0, 20.0)))

    def torso_world_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Return torso angular (rad/s) and linear (m/s) velocity in world coordinates."""
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.torso_body,
            velocity,
            0,
        )
        return velocity[:3].copy(), velocity[3:].copy()

    def torso_tilt_angle(self) -> float:
        """Return the unsigned torso tilt from world vertical in radians."""
        return float(np.arccos(np.clip(self.torso_up(), -1.0, 1.0)))

    def torso_tilt_components(self) -> tuple[float, float]:
        """Return signed lateral and sagittal torso lean angles in radians.

        The robot's reset orientation makes world X the observed lateral
        balance direction and world Y the sagittal direction for this task.
        Positive/negative signs identify the lean direction; penalties use
        their squared magnitudes and are therefore direction symmetric.
        """
        torso_rot = self.data.xmat[self.torso_body].reshape(3, 3)
        torso_up_axis = torso_rot[:, 2]
        lateral = np.arctan2(torso_up_axis[0], torso_up_axis[2])
        sagittal = np.arctan2(torso_up_axis[1], torso_up_axis[2])
        return float(lateral), float(sagittal)

    def torso_tilt_rate(self) -> float:
        """Return d(tilt angle)/dt; positive values mean the torso is tipping farther."""
        torso_rot = self.data.xmat[self.torso_body].reshape(3, 3)
        torso_up_axis = torso_rot[:, 2]
        angular_velocity, _ = self.torso_world_velocity()
        up_axis_velocity = np.cross(angular_velocity, torso_up_axis)
        sin_theta = float(np.sqrt(max(1.0 - torso_up_axis[2] ** 2, 0.0)))
        if sin_theta < 1e-6:
            return 0.0
        return float(-up_axis_velocity[2] / sin_theta)

    def torso_downward_speed(self) -> float:
        """Return only the downward component of torso world vertical speed in m/s."""
        _, linear_velocity = self.torso_world_velocity()
        return float(max(0.0, -linear_velocity[2]))

    def torso_up(self) -> float:
        return float(self.data.xmat[self.torso_body].reshape(3, 3)[2, 2])

    def board_up(self) -> float:
        return float(self.data.xmat[self.board_body].reshape(3, 3)[2, 2])

    def pelvis_height(self) -> float:
        return float(self.data.xpos[self.pelvis_body, 2])

    def board_roller_xy_error(self) -> float:
        return float(np.linalg.norm(self.data.xpos[self.board_body, :2] - self.data.xpos[self.roller_body, :2]))

    def _joint_qpos_id(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.model.jnt_qposadr[jid])

    def _joint_dof_id(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.model.jnt_dofadr[jid])

    def _body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _site_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)

    def _actuator_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def _actuator_name(self, actuator_id: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)

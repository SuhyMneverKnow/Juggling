"""Environment scaffold for the H1-2 balance-board mounting task.

Reset dynamics, success, and termination are defined. The task-specific
observation and reward will be added next.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from envs.balance_env import ACTION_SCALES, CONTROLLED_JOINTS
from rewards.mount_rewards import mount_reward


PROJECT_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = PROJECT_ROOT / "models" / "mount.xml"

INIT_JOINT_POS = {
    # Symmetric 8-degree knee flexion, with hip/ankle compensation that keeps
    # the torso upright and both soles level.
    "left_hip_pitch_joint": -0.07,
    "left_knee_joint": 0.14,
    "left_ankle_pitch_joint": -0.07,
    "right_hip_pitch_joint": -0.07,
    "right_knee_joint": 0.14,
    "right_ankle_pitch_joint": -0.07,
    "left_elbow_joint": np.pi / 2.0,
    "right_elbow_joint": np.pi / 2.0,
}

# A successful mount must be a stable hand-off state for the Part 2 balance
# policy, not merely a transient foot contact with the board.
SUCCESS_FOOT_SAFE_X = 0.30  # m, board-local half-length allowance
SUCCESS_FOOT_SAFE_Y = 0.04  # m, board-local half-width allowance
SUCCESS_FOOT_Z_MIN = 0.0  # m, site height relative to board centre
SUCCESS_FOOT_Z_MAX = 0.05
SUCCESS_PELVIS_Z_MIN = 1.10  # m, world frame
SUCCESS_PELVIS_ABOVE_BOARD_MIN = 0.98  # m
SUCCESS_TORSO_TILT_MAX = np.deg2rad(20.0)
SUCCESS_PELVIS_VERTICAL_SPEED_MAX = 0.30  # m/s
SUCCESS_TORSO_ANGULAR_SPEED_MAX = 1.0  # rad/s
SUCCESS_BOARD_TILT_MAX = np.deg2rad(30.0)
SUCCESS_BOARD_ANGULAR_SPEED_MAX = 1.0  # rad/s
SUCCESS_BOARD_ROLLER_HORIZONTAL_ERROR_MAX = 0.15  # m
SUCCESS_ROLLER_AXIS_ERROR_MAX = np.deg2rad(20.0)
SUCCESS_HOLD_SECONDS = 0.30

# Desired hand-off foot sites in board-local coordinates. These correspond to
# the centred Part 2 standing pose validated against the current geometry.
LEFT_FOOT_TARGET_IN_BOARD = np.array([-0.163, 0.0, 0.0158], dtype=np.float64)
RIGHT_FOOT_TARGET_IN_BOARD = np.array([0.163, 0.0, 0.0158], dtype=np.float64)
OBSERVATION_DIM = 102

# Failure thresholds deliberately leave a recovery band outside the success
# region. Time-limit exhaustion is handled separately as truncation.
FAILURE_PELVIS_Z_MIN = 0.55  # m
FAILURE_TORSO_TILT_MAX = np.deg2rad(70.0)
FAILURE_BOARD_TILT_MAX = np.deg2rad(70.0)
FAILURE_BOARD_ROLLER_HORIZONTAL_ERROR_MAX = 0.35  # m
FAILURE_ROLLER_AXIS_ERROR_MAX = np.deg2rad(60.0)
FAILURE_BOARD_ROLLER_LOST_CONTACT_SECONDS = 0.20
FAILURE_ROBOT_BOARD_HORIZONTAL_DISTANCE_MAX = 1.20  # m
FAILURE_ABS_QVEL_MAX = 100.0


class MountEnv(gym.Env):
    """Mount-task environment with reset, success, and failure semantics."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        episode_seconds: float = 4.0,
        control_dt: float = 0.02,
        reset_position_noise: float = 0.01,
        reset_velocity_noise: float = 0.02,
    ) -> None:
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None
        self.control_dt = control_dt
        self.frame_skip = max(1, int(round(control_dt / self.model.opt.timestep)))
        self.max_steps = max(1, int(round(episode_seconds / control_dt)))
        self.board_roller_lost_contact_steps_max = max(
            1,
            int(
                round(
                    FAILURE_BOARD_ROLLER_LOST_CONTACT_SECONDS / control_dt
                )
            ),
        )
        self.reset_position_noise = reset_position_noise
        self.reset_velocity_noise = reset_velocity_noise
        self.action_scales = ACTION_SCALES.copy()
        self.success_hold_steps_required = max(
            1, int(round(SUCCESS_HOLD_SECONDS / control_dt))
        )
        self.success_hold_steps = 0
        self.board_roller_lost_contact_steps = 0
        self.step_count = 0

        self.init_qpos = self.model.qpos0.copy()
        for joint_name, joint_pos in INIT_JOINT_POS.items():
            self.init_qpos[self._joint_qpos_id(joint_name)] = joint_pos
        self.init_qvel = np.zeros(self.model.nv, dtype=np.float64)

        self.actuator_ids = np.array(
            [self._actuator_id(name) for name in CONTROLLED_JOINTS], dtype=np.int32
        )
        self.qpos_ids = np.array(
            [self._joint_qpos_id(name) for name in CONTROLLED_JOINTS], dtype=np.int32
        )
        self.qvel_ids = np.array(
            [self._joint_dof_id(name) for name in CONTROLLED_JOINTS], dtype=np.int32
        )
        self.default_joint_pos = self.init_qpos[self.qpos_ids].copy()

        self.pelvis_body = self._body_id("pelvis")
        self.torso_body = self._body_id("torso_link")
        self.board_body = self._body_id("board")
        self.roller_body = self._body_id("roller")
        self.left_foot_site = self._site_id("left_foot")
        self.right_foot_site = self._site_id("right_foot")
        self.floating_base_dof = self._joint_dof_id("floating_base_joint")
        self.board_free_dof = self._joint_dof_id("board_free")
        self.roller_free_dof = self._joint_dof_id("roller_free")

        self.floor_geom = self._geom_id("floor")
        self.board_geom = self._geom_id("board_geom")
        self.roller_geom = self._geom_id("roller_geom")
        self.left_foot_geom = self._geom_id("left_foot_collision")
        self.right_foot_geom = self._geom_id("right_foot_collision")
        self.robot_body_ids = self._body_descendants(self.pelvis_body)

        self.all_actuator_joint_qpos = np.array(
            [self._joint_qpos_id(self._actuator_name(i)) for i in range(self.model.nu)],
            dtype=np.int32,
        )
        self.all_actuator_joint_dof = np.array(
            [self._joint_dof_id(self._actuator_name(i)) for i in range(self.model.nu)],
            dtype=np.int32,
        )
        self.all_default_joint_pos = self.init_qpos[self.all_actuator_joint_qpos].copy()
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()

        # Match Part 2 exactly so the reset comparison is not confounded by a
        # different low-level controller.
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
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(len(CONTROLLED_JOINTS),), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(OBSERVATION_DIM,), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        self.last_action[:] = 0.0
        self.success_hold_steps = 0
        self.board_roller_lost_contact_steps = 0
        self.step_count = 0

        if self.reset_position_noise > 0.0:
            self.data.qpos[self.qpos_ids] += self.np_random.uniform(
                -self.reset_position_noise,
                self.reset_position_noise,
                size=len(self.qpos_ids),
            )
        if self.reset_velocity_noise > 0.0:
            self.data.qvel[self.qvel_ids] += self.np_random.uniform(
                -self.reset_velocity_noise,
                self.reset_velocity_noise,
                size=len(self.qvel_ids),
            )

        mujoco.mj_forward(self.model, self.data)
        self._apply_pd(np.zeros_like(self.last_action))
        info = self.task_diagnostics()
        return self._get_obs(), info

    def step(self, action):
        """Advance the dynamics; the task reward is intentionally still zero."""
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        previous_action = self.last_action.copy()
        for _ in range(self.frame_skip):
            self._apply_pd(action)
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        self.last_action = action.astype(np.float32)

        if self._geoms_touching(self.board_geom, self.roller_geom):
            self.board_roller_lost_contact_steps = 0
        else:
            self.board_roller_lost_contact_steps += 1

        conditions = self.success_conditions()
        if conditions["success_conditions_now"]:
            self.success_hold_steps += 1
        else:
            self.success_hold_steps = 0
        is_success = self.success_hold_steps >= self.success_hold_steps_required
        failures = self.failure_conditions(conditions)
        terminated = bool(is_success or failures["physical_failure"])
        timeout = bool(self.step_count >= self.max_steps and not terminated)
        truncated = timeout

        if is_success:
            termination_reason = "success"
        elif failures["numerical_failure"]:
            termination_reason = "numerical_failure"
        elif failures["robot_fallen"]:
            termination_reason = "robot_fallen"
        elif failures["board_roller_failed"]:
            termination_reason = "board_roller_failed"
        elif failures["out_of_area"]:
            termination_reason = "out_of_area"
        elif timeout:
            termination_reason = "timeout"
        else:
            termination_reason = "none"

        info = self.task_diagnostics(conditions)
        info.update(failures)
        info["success_hold_steps"] = self.success_hold_steps
        info["success_hold_seconds"] = self.success_hold_steps * self.control_dt
        info["board_roller_lost_contact_steps"] = self.board_roller_lost_contact_steps
        info["board_roller_lost_contact_seconds"] = (
            self.board_roller_lost_contact_steps * self.control_dt
        )
        info["episode_step"] = self.step_count
        info["episode_seconds"] = self.step_count * self.control_dt
        info["is_success"] = is_success
        info["timeout"] = timeout
        info["termination_reason"] = termination_reason
        reward, reward_info = mount_reward(self, action, previous_action, info)
        reward = float(np.nan_to_num(reward, nan=-10.0, posinf=10.0, neginf=-10.0))
        info.update(reward_info)
        return self._get_obs(), reward, terminated, truncated, info

    def failure_conditions(
        self, conditions: dict[str, float | bool] | None = None
    ) -> dict[str, float | bool]:
        if conditions is None:
            conditions = self.success_conditions()

        non_foot_body_on_floor = self._has_non_foot_floor_contact()
        pelvis_too_low = float(conditions["pelvis_z"]) < FAILURE_PELVIS_Z_MIN
        torso_severely_tilted = (
            float(conditions["torso_tilt_rad"]) > FAILURE_TORSO_TILT_MAX
        )
        robot_fallen = bool(
            non_foot_body_on_floor or pelvis_too_low or torso_severely_tilted
        )

        board_roller_too_far = (
            float(conditions["board_roller_horizontal_error"])
            > FAILURE_BOARD_ROLLER_HORIZONTAL_ERROR_MAX
        )
        roller_axis_failed = (
            float(conditions["roller_axis_error_rad"])
            > FAILURE_ROLLER_AXIS_ERROR_MAX
        )
        board_severely_tilted = (
            float(conditions["board_tilt_rad"]) > FAILURE_BOARD_TILT_MAX
        )
        board_roller_contact_lost = (
            self.board_roller_lost_contact_steps
            >= self.board_roller_lost_contact_steps_max
        )
        board_roller_failed = bool(
            board_roller_too_far
            or roller_axis_failed
            or board_severely_tilted
            or board_roller_contact_lost
        )

        pelvis_xy = self.data.xpos[self.pelvis_body, :2]
        board_xy = self.data.xpos[self.board_body, :2]
        robot_board_horizontal_distance = float(np.linalg.norm(pelvis_xy - board_xy))
        out_of_area = bool(
            robot_board_horizontal_distance
            > FAILURE_ROBOT_BOARD_HORIZONTAL_DISTANCE_MAX
        )

        state_finite = bool(
            np.isfinite(self.data.qpos).all()
            and np.isfinite(self.data.qvel).all()
        )
        max_abs_qvel = float(
            np.max(np.abs(self.data.qvel)) if self.data.qvel.size else 0.0
        )
        numerical_failure = bool(
            not state_finite or max_abs_qvel > FAILURE_ABS_QVEL_MAX
        )
        physical_failure = bool(
            robot_fallen or board_roller_failed or out_of_area or numerical_failure
        )
        return {
            "physical_failure": physical_failure,
            "robot_fallen": robot_fallen,
            "non_foot_body_on_floor": non_foot_body_on_floor,
            "pelvis_too_low": pelvis_too_low,
            "torso_severely_tilted": torso_severely_tilted,
            "board_roller_failed": board_roller_failed,
            "board_roller_too_far": board_roller_too_far,
            "roller_axis_failed": roller_axis_failed,
            "board_severely_tilted": board_severely_tilted,
            "board_roller_contact_lost": board_roller_contact_lost,
            "out_of_area": out_of_area,
            "robot_board_horizontal_distance": robot_board_horizontal_distance,
            "numerical_failure": numerical_failure,
            "state_finite": state_finite,
            "max_abs_qvel": max_abs_qvel,
        }

    def success_conditions(self) -> dict[str, float | bool]:
        left_foot_board = self._site_pos_in_board(self.left_foot_site)
        right_foot_board = self._site_pos_in_board(self.right_foot_site)
        contacts = [
            {int(self.data.contact[i].geom1), int(self.data.contact[i].geom2)}
            for i in range(self.data.ncon)
        ]

        def touching(geom_a: int, geom_b: int) -> bool:
            return any({geom_a, geom_b} <= pair for pair in contacts)

        left_foot_on_floor = touching(self.left_foot_geom, self.floor_geom)
        right_foot_on_floor = touching(self.right_foot_geom, self.floor_geom)
        left_foot_on_board = touching(self.left_foot_geom, self.board_geom)
        right_foot_on_board = touching(self.right_foot_geom, self.board_geom)
        left_foot_safe = self._foot_in_safe_region(left_foot_board)
        right_foot_safe = self._foot_in_safe_region(right_foot_board)

        pelvis_z = float(self.data.xpos[self.pelvis_body, 2])
        board_z = float(self.data.xpos[self.board_body, 2])
        pelvis_above_board = pelvis_z - board_z
        pelvis_vertical_speed = float(
            self.data.qvel[self.floating_base_dof + 2]
        )
        torso_tilt = self._body_tilt(self.torso_body)
        torso_angular_speed = float(
            np.linalg.norm(self.data.cvel[self.torso_body, :3])
        )
        board_tilt = self._body_tilt(self.board_body)
        board_angular_speed = float(
            np.linalg.norm(self.data.qvel[self.board_free_dof + 3 : self.board_free_dof + 6])
        )

        roller_pos_in_board = self._point_in_board(
            self.data.xpos[self.roller_body]
        )
        board_roller_horizontal_error = float(
            np.linalg.norm(roller_pos_in_board[:2])
        )
        board_y_axis = self.data.xmat[self.board_body].reshape(3, 3)[:, 1]
        roller_axis = self.data.xmat[self.roller_body].reshape(3, 3)[:, 2]
        axis_alignment = float(np.clip(abs(np.dot(board_y_axis, roller_axis)), 0.0, 1.0))
        roller_axis_error = float(np.arccos(axis_alignment))
        no_illegal_body_contact = not self._has_illegal_body_contact()

        checks = {
            "both_feet_on_board": left_foot_on_board and right_foot_on_board,
            "both_feet_off_floor": not left_foot_on_floor and not right_foot_on_floor,
            "both_feet_in_safe_region": left_foot_safe and right_foot_safe,
            "no_illegal_body_contact": no_illegal_body_contact,
            "pelvis_world_height_ok": pelvis_z > SUCCESS_PELVIS_Z_MIN,
            "pelvis_board_height_ok": pelvis_above_board > SUCCESS_PELVIS_ABOVE_BOARD_MIN,
            "torso_tilt_ok": torso_tilt < SUCCESS_TORSO_TILT_MAX,
            "pelvis_vertical_speed_ok": abs(pelvis_vertical_speed)
            < SUCCESS_PELVIS_VERTICAL_SPEED_MAX,
            "torso_angular_speed_ok": torso_angular_speed
            < SUCCESS_TORSO_ANGULAR_SPEED_MAX,
            "board_tilt_ok": board_tilt < SUCCESS_BOARD_TILT_MAX,
            "board_angular_speed_ok": board_angular_speed
            < SUCCESS_BOARD_ANGULAR_SPEED_MAX,
            "board_roller_position_ok": board_roller_horizontal_error
            < SUCCESS_BOARD_ROLLER_HORIZONTAL_ERROR_MAX,
            "roller_axis_ok": roller_axis_error < SUCCESS_ROLLER_AXIS_ERROR_MAX,
        }
        return {
            **checks,
            "success_conditions_now": all(checks.values()),
            "left_foot_on_floor": left_foot_on_floor,
            "right_foot_on_floor": right_foot_on_floor,
            "left_foot_on_board": left_foot_on_board,
            "right_foot_on_board": right_foot_on_board,
            "left_foot_safe": left_foot_safe,
            "right_foot_safe": right_foot_safe,
            "left_foot_board_x": float(left_foot_board[0]),
            "left_foot_board_y": float(left_foot_board[1]),
            "left_foot_board_z": float(left_foot_board[2]),
            "right_foot_board_x": float(right_foot_board[0]),
            "right_foot_board_y": float(right_foot_board[1]),
            "right_foot_board_z": float(right_foot_board[2]),
            "pelvis_z": pelvis_z,
            "pelvis_above_board": pelvis_above_board,
            "pelvis_vertical_speed": pelvis_vertical_speed,
            "torso_tilt_rad": torso_tilt,
            "torso_tilt_deg": float(np.rad2deg(torso_tilt)),
            "torso_angular_speed": torso_angular_speed,
            "board_tilt_rad": board_tilt,
            "board_tilt_deg": float(np.rad2deg(board_tilt)),
            "board_angular_speed": board_angular_speed,
            "board_roller_horizontal_error": board_roller_horizontal_error,
            "roller_axis_error_rad": roller_axis_error,
            "roller_axis_error_deg": float(np.rad2deg(roller_axis_error)),
        }

    def task_diagnostics(
        self, conditions: dict[str, float | bool] | None = None
    ) -> dict[str, float | bool]:
        if conditions is None:
            conditions = self.success_conditions()
        return {
            **conditions,
            "pelvis_x": float(self.data.xpos[self.pelvis_body, 0]),
            "pelvis_y": float(self.data.xpos[self.pelvis_body, 1]),
            "board_x": float(self.data.xpos[self.board_body, 0]),
            "board_y": float(self.data.xpos[self.board_body, 1]),
            "board_z": float(self.data.xpos[self.board_body, 2]),
            "roller_x": float(self.data.xpos[self.roller_body, 0]),
            "roller_y": float(self.data.xpos[self.roller_body, 1]),
            "roller_z": float(self.data.xpos[self.roller_body, 2]),
        }

    # Backward-compatible name used by the reset viewer.
    def reset_diagnostics(self) -> dict[str, float | bool]:
        return self.task_diagnostics()

    def render(self):
        if self.render_mode != "human":
            return None
        if self.viewer is None:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _get_obs(self) -> np.ndarray:
        pelvis_rotation = self.data.xmat[self.pelvis_body].reshape(3, 3)
        board_rotation = self.data.xmat[self.board_body].reshape(3, 3)
        heading_rotation = self._heading_rotation(pelvis_rotation)

        pelvis_angular_world, pelvis_linear_world = self._object_velocity(
            mujoco.mjtObj.mjOBJ_BODY, self.pelvis_body
        )
        board_angular_world, board_linear_world = self._object_velocity(
            mujoco.mjtObj.mjOBJ_BODY, self.board_body
        )
        roller_angular_world, roller_linear_world = self._object_velocity(
            mujoco.mjtObj.mjOBJ_BODY, self.roller_body
        )
        _, left_foot_linear_world = self._object_velocity(
            mujoco.mjtObj.mjOBJ_SITE, self.left_foot_site
        )
        _, right_foot_linear_world = self._object_velocity(
            mujoco.mjtObj.mjOBJ_SITE, self.right_foot_site
        )

        projected_gravity = pelvis_rotation.T @ np.array([0.0, 0.0, -1.0])
        pelvis_linear_local = pelvis_rotation.T @ pelvis_linear_world
        pelvis_angular_local = pelvis_rotation.T @ pelvis_angular_world

        board_position_relative = heading_rotation.T @ (
            self.data.xpos[self.board_body] - self.data.xpos[self.pelvis_body]
        )
        board_long_axis = heading_rotation.T @ board_rotation[:, 0]
        board_up_axis = heading_rotation.T @ board_rotation[:, 2]
        board_linear_relative = heading_rotation.T @ (
            board_linear_world - pelvis_linear_world
        )
        board_angular_relative = heading_rotation.T @ (
            board_angular_world - pelvis_angular_world
        )

        roller_position_in_board = self._point_in_board(
            self.data.xpos[self.roller_body]
        )
        roller_axis_in_board = board_rotation.T @ (
            self.data.xmat[self.roller_body].reshape(3, 3)[:, 2]
        )
        roller_linear_relative = board_rotation.T @ (
            roller_linear_world
            - board_linear_world
            - np.cross(
                board_angular_world,
                self.data.xpos[self.roller_body] - self.data.xpos[self.board_body],
            )
        )
        roller_angular_relative = board_rotation.T @ (
            roller_angular_world - board_angular_world
        )

        left_foot_in_board = self._site_pos_in_board(self.left_foot_site)
        right_foot_in_board = self._site_pos_in_board(self.right_foot_site)
        left_foot_target_error = left_foot_in_board - LEFT_FOOT_TARGET_IN_BOARD
        right_foot_target_error = right_foot_in_board - RIGHT_FOOT_TARGET_IN_BOARD
        left_foot_velocity_in_board = self._point_velocity_relative_to_board(
            self.data.site_xpos[self.left_foot_site], left_foot_linear_world
        )
        right_foot_velocity_in_board = self._point_velocity_relative_to_board(
            self.data.site_xpos[self.right_foot_site], right_foot_linear_world
        )

        contact_flags = np.array(
            [
                self._geoms_touching(self.left_foot_geom, self.floor_geom),
                self._geoms_touching(self.right_foot_geom, self.floor_geom),
                self._geoms_touching(self.left_foot_geom, self.board_geom),
                self._geoms_touching(self.right_foot_geom, self.board_geom),
            ],
            dtype=np.float64,
        )
        episode_progress = np.array(
            [min(self.step_count / self.max_steps, 1.0)], dtype=np.float64
        )

        obs = np.concatenate(
            [
                self.data.qpos[self.qpos_ids] - self.default_joint_pos,  # 16
                self.data.qvel[self.qvel_ids],  # 16
                projected_gravity,  # 3
                pelvis_linear_local,  # 3
                pelvis_angular_local,  # 3
                np.array([self.data.xpos[self.pelvis_body, 2]]),  # 1
                self.last_action,  # 16
                board_position_relative,  # 3
                board_long_axis,  # 3
                board_up_axis,  # 3
                board_linear_relative,  # 3
                board_angular_relative,  # 3
                roller_position_in_board,  # 3
                roller_axis_in_board,  # 3
                roller_linear_relative,  # 3
                roller_angular_relative,  # 3
                left_foot_target_error,  # 3
                right_foot_target_error,  # 3
                left_foot_velocity_in_board,  # 3
                right_foot_velocity_in_board,  # 3
                contact_flags,  # 4
                episode_progress,  # 1
            ]
        )
        if obs.shape != (OBSERVATION_DIM,):
            raise RuntimeError(
                f"Mount observation shape {obs.shape} does not match "
                f"({OBSERVATION_DIM},)."
            )
        return np.nan_to_num(
            obs, nan=0.0, posinf=10.0, neginf=-10.0
        ).astype(np.float32)

    @staticmethod
    def _heading_rotation(pelvis_rotation: np.ndarray) -> np.ndarray:
        """Return a yaw-only robot heading frame as a world rotation matrix."""
        forward = pelvis_rotation[:, 0].copy()
        forward[2] = 0.0
        norm = np.linalg.norm(forward)
        if norm < 1e-8:
            forward = np.array([1.0, 0.0, 0.0])
        else:
            forward /= norm
        left = np.array([-forward[1], forward[0], 0.0])
        return np.column_stack((forward, left, np.array([0.0, 0.0, 1.0])))

    def _object_velocity(
        self, object_type: mujoco.mjtObj, object_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model, self.data, object_type, object_id, velocity, 0
        )
        return velocity[:3], velocity[3:]

    def _point_velocity_relative_to_board(
        self, point_world: np.ndarray, point_velocity_world: np.ndarray
    ) -> np.ndarray:
        board_angular_world, board_linear_world = self._object_velocity(
            mujoco.mjtObj.mjOBJ_BODY, self.board_body
        )
        board_velocity_at_point = board_linear_world + np.cross(
            board_angular_world,
            np.asarray(point_world) - self.data.xpos[self.board_body],
        )
        board_rotation = self.data.xmat[self.board_body].reshape(3, 3)
        return board_rotation.T @ (
            np.asarray(point_velocity_world) - board_velocity_at_point
        )

    def _point_in_board(self, point_world: np.ndarray) -> np.ndarray:
        board_rotation = self.data.xmat[self.board_body].reshape(3, 3)
        return board_rotation.T @ (
            np.asarray(point_world) - self.data.xpos[self.board_body]
        )

    def _site_pos_in_board(self, site_id: int) -> np.ndarray:
        return self._point_in_board(self.data.site_xpos[site_id])

    @staticmethod
    def _foot_in_safe_region(foot_board: np.ndarray) -> bool:
        return bool(
            abs(foot_board[0]) < SUCCESS_FOOT_SAFE_X
            and abs(foot_board[1]) < SUCCESS_FOOT_SAFE_Y
            and SUCCESS_FOOT_Z_MIN < foot_board[2] < SUCCESS_FOOT_Z_MAX
        )

    def _body_tilt(self, body_id: int) -> float:
        body_up = self.data.xmat[body_id].reshape(3, 3)[:, 2]
        return float(np.arccos(np.clip(body_up[2], -1.0, 1.0)))

    def _has_illegal_body_contact(self) -> bool:
        external_geoms = {self.floor_geom, self.board_geom, self.roller_geom}
        legal_feet = {self.left_foot_geom, self.right_foot_geom}
        for i in range(self.data.ncon):
            geom1 = int(self.data.contact[i].geom1)
            geom2 = int(self.data.contact[i].geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            if body1 in self.robot_body_ids and geom2 in external_geoms:
                if geom1 not in legal_feet:
                    return True
            if body2 in self.robot_body_ids and geom1 in external_geoms:
                if geom2 not in legal_feet:
                    return True
        return False

    def _has_non_foot_floor_contact(self) -> bool:
        legal_feet = {self.left_foot_geom, self.right_foot_geom}
        for i in range(self.data.ncon):
            geom1 = int(self.data.contact[i].geom1)
            geom2 = int(self.data.contact[i].geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            if geom2 == self.floor_geom and body1 in self.robot_body_ids:
                if geom1 not in legal_feet:
                    return True
            if geom1 == self.floor_geom and body2 in self.robot_body_ids:
                if geom2 not in legal_feet:
                    return True
        return False

    def _geoms_touching(self, geom_a: int, geom_b: int) -> bool:
        for i in range(self.data.ncon):
            pair = {
                int(self.data.contact[i].geom1),
                int(self.data.contact[i].geom2),
            }
            if {geom_a, geom_b} <= pair:
                return True
        return False

    def _apply_pd(self, action: np.ndarray) -> None:
        targets = self.all_default_joint_pos.copy()
        targets[self.actuator_ids] = (
            self.default_joint_pos + self.action_scales * np.asarray(action)
        )
        torque = self.kp * (targets - self.data.qpos[self.all_actuator_joint_qpos])
        torque -= self.kd * self.data.qvel[self.all_actuator_joint_dof]
        self.data.ctrl[:] = np.clip(torque, self.ctrl_low, self.ctrl_high)

    def _body_id(self, name: str) -> int:
        return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name))

    def _joint_qpos_id(self, name: str) -> int:
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.model.jnt_qposadr[joint])

    def _geom_id(self, name: str) -> int:
        return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name))

    def _site_id(self, name: str) -> int:
        return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name))

    def _body_descendants(self, root_body: int) -> set[int]:
        descendants = {root_body}
        changed = True
        while changed:
            changed = False
            for body_id, parent_id in enumerate(self.model.body_parentid):
                if int(parent_id) in descendants and body_id not in descendants:
                    descendants.add(body_id)
                    changed = True
        return descendants

    def _joint_dof_id(self, name: str) -> int:
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.model.jnt_dofadr[joint])

    def _actuator_id(self, name: str) -> int:
        return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))

    def _actuator_name(self, actuator_id: int) -> str:
        name = mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
        )
        if name is None:
            raise KeyError(f"Actuator {actuator_id} has no name")
        return name

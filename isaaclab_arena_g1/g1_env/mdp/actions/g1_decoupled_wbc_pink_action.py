# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import numpy as np
import torch
from collections.abc import Sequence
from scipy.spatial.transform import Rotation as R
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation

from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_joint_action import G1DecoupledWBCJointAction
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.g1_wbc_upperbody_ik.g1_wbc_upperbody_controller import (
    G1WBCUpperbodyController,
)
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.action_constants import (
    ACTION_DIM_WITH_BILATERAL_FINGERS,
    ACTION_DIM_WITH_RIGHT_FINGERS,
    BASE_ACTION_DIM,
    BASE_HEIGHT_CMD_END_IDX,
    BASE_HEIGHT_CMD_START_IDX,
    BILATERAL_RIGHT_FINGER_ANGLES_END_IDX,
    BILATERAL_RIGHT_FINGER_ANGLES_START_IDX,
    LEFT_HAND_STATE_DIM,
    LEFT_HAND_STATE_IDX,
    LEFT_FINGER_ANGLES_END_IDX,
    LEFT_FINGER_ANGLES_START_IDX,
    LEFT_WRIST_LINK_NAME,
    LEFT_WRIST_POS_DIM,
    LEFT_WRIST_POS_END_IDX,
    LEFT_WRIST_POS_START_IDX,
    LEFT_WRIST_QUAT_DIM,
    LEFT_WRIST_QUAT_END_IDX,
    LEFT_WRIST_QUAT_START_IDX,
    NAVIGATE_CMD_END_IDX,
    NAVIGATE_CMD_START_IDX,
    NAVIGATE_THRESHOLD,
    RIGHT_FINGER_ANGLES_END_IDX,
    RIGHT_FINGER_ANGLES_START_IDX,
    RIGHT_HAND_STATE_DIM,
    RIGHT_HAND_STATE_IDX,
    RIGHT_WRIST_LINK_NAME,
    RIGHT_WRIST_POS_DIM,
    RIGHT_WRIST_POS_END_IDX,
    RIGHT_WRIST_POS_START_IDX,
    RIGHT_WRIST_QUAT_DIM,
    RIGHT_WRIST_QUAT_END_IDX,
    RIGHT_WRIST_QUAT_START_IDX,
    TORSO_ORIENTATION_RPY_CMD_END_IDX,
    TORSO_ORIENTATION_RPY_CMD_START_IDX,
)
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.run_policy import postprocess_actions, prepare_observations
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.utils.p_controller import PController

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.embodiments.g1.mdp.actions.g1_decoupled_wbc_pink_action_cfg import G1DecoupledWBCPinkActionCfg


class G1DecoupledWBCPinkAction(G1DecoupledWBCJointAction):
    """Action term for the G1 decoupled WBC policy. Upper body PINK IK control, lower body RL-based policy."""

    cfg: G1DecoupledWBCPinkActionCfg

    _asset: Articulation
    """The articulation asset to which the action term is applied."""

    def __init__(self, cfg: G1DecoupledWBCPinkActionCfg, env: ManagerBasedEnv):
        """Initialize the action term.

        Args:
            cfg: The configuration for this action term.
            env: The environment in which the action term will be applied.
        """
        super().__init__(cfg, env)

        assert self.num_envs == 1, "PINK controller currently only supports single environment"

        self.navigation_p_controller = PController(
            distance_error_threshold=self.cfg.distance_error_threshold,
            heading_diff_threshold=self.cfg.heading_diff_threshold,
            kp_angular_turning_only=self.cfg.kp_angular_turning_only,
            kp_linear_x=self.cfg.kp_linear_x,
            kp_linear_y=self.cfg.kp_linear_y,
            kp_angular=self.cfg.kp_angular,
            min_vel=self.cfg.min_vel,
            max_vel=self.cfg.max_vel,
            num_envs=self.num_envs,
            inplace_turning_flag=self.cfg.turning_first,
        )

        # Mimic navigation P-controller variables
        self._is_navigating = False
        self._navigation_goal_reached = False
        self._navigation_step_counter = 0
        self._num_navigation_subgoals_reached = -1
        self._navigate_cmd = torch.zeros([self.num_envs, 3], device=self.device)
        self._torso_orientation_rpy_cmd = torch.zeros([self.num_envs, 3], device=self.device)

        # Create the PINK IK controller
        self.upperbody_controller = G1WBCUpperbodyController(
            robot_model=self.robot_model,
            body_active_joint_groups=["arms"],
        )
        self._upper_body_joint_indices = self.robot_model.get_joint_group_indices("upper_body")
        self._full_index_to_joint_name = {int(v): k for k, v in self.robot_model.joint_to_dof_index.items()}
        nav_arm_enable_raw = str(os.environ.get("G1_NAV_STRAIGHT_ARM_ENABLE", "1")).strip().lower()
        self._nav_straight_arm_enable = nav_arm_enable_raw not in {"0", "false", "off", "no"}
        try:
            self._nav_straight_arm_speed_threshold = float(os.environ.get("G1_NAV_STRAIGHT_ARM_SPEED_THRESH", "0.02"))
        except ValueError:
            self._nav_straight_arm_speed_threshold = 0.02
        self._nav_straight_arm_targets = self._build_nav_straight_arm_targets()
        print(
            "[g1][nav_straight_arm] "
            f"enable={self._nav_straight_arm_enable}, speed_thresh={self._nav_straight_arm_speed_threshold:.4f}"
        )

    # Properties.
    # """
    @property
    def is_navigating(self) -> bool:
        """Get the is navigating flag."""
        return self._is_navigating

    @property
    def navigation_goal_reached(self) -> bool:
        """Get the navigation goal reached tensor."""
        return self._navigation_goal_reached

    @property
    def left_wrist_pos_dim(self) -> int:
        """Dimension of left wrist position command."""
        return LEFT_WRIST_POS_DIM

    @property
    def left_wrist_quat_dim(self) -> int:
        """Dimension of left wrist quaternion command."""
        return LEFT_WRIST_QUAT_DIM

    @property
    def right_wrist_pos_dim(self) -> int:
        """Dimension of right wrist position command."""
        return RIGHT_WRIST_POS_DIM

    @property
    def right_wrist_quat_dim(self) -> int:
        """Dimension of right wrist quaternion command."""
        return RIGHT_WRIST_QUAT_DIM

    @property
    def left_hand_state_dim(self) -> int:
        """Dimension of left hand state command."""
        return LEFT_HAND_STATE_DIM

    @property
    def right_hand_state_dim(self) -> int:
        """Dimension of right hand state command."""
        return RIGHT_HAND_STATE_DIM

    @property
    def action_dim(self) -> int:
        """Dimension of the action space."""
        return (
            self.left_hand_state_dim
            + self.right_hand_state_dim
            + self.left_wrist_pos_dim
            + self.left_wrist_quat_dim
            + self.right_wrist_pos_dim
            + self.right_wrist_quat_dim
            + self.navigate_cmd_dim
            + self.base_height_cmd_dim
            + self.torso_orientation_rpy_cmd_dim
        )

    @property
    def raw_actions(self) -> torch.Tensor:
        """Get the raw actions tensor."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Get the processed actions tensor."""
        return self._processed_actions

    @property
    def navigate_cmd(self):
        return self._navigate_cmd

    def get_navigation_cmd_from_actions(self, actions: torch.Tensor):
        """Get navigation command from fixed 23D WBC-PINK action layout."""
        return actions[:, NAVIGATE_CMD_START_IDX:NAVIGATE_CMD_END_IDX]

    def get_base_height_cmd_from_actions(self, actions: torch.Tensor):
        """Get base height command from fixed 23D WBC-PINK action layout."""
        return actions[:, BASE_HEIGHT_CMD_START_IDX:BASE_HEIGHT_CMD_END_IDX]

    def get_torso_orientation_rpy_cmd_from_actions(self, actions: torch.Tensor):
        """Get torso orientation command from fixed 23D WBC-PINK action layout."""
        return actions[:, TORSO_ORIENTATION_RPY_CMD_START_IDX:TORSO_ORIENTATION_RPY_CMD_END_IDX]

    def compute_upperbody_joint_positions(
        self,
        body_data: dict[str, np.ndarray],
        left_hand_state: np.ndarray | torch.Tensor,
        right_hand_state: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Run the PINK IK controller to compute the target joint positions for the upper body."""
        if self.upperbody_controller.in_warmup:
            for _ in range(50):
                target_robot_joints = self.upperbody_controller.inverse_kinematics(
                    body_data, left_hand_state, right_hand_state
                )
            self.upperbody_controller.in_warmup = False
        else:
            target_robot_joints = self.upperbody_controller.inverse_kinematics(
                body_data, left_hand_state, right_hand_state
            )
        return target_robot_joints

    def _extract_hand_states_from_actions(
        self, actions: torch.Tensor
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        """Extract hand commands from actions with backward-compatible layouts.

        Supported layouts:
        - 23D base action: scalar left/right hand states
        - 35D action: base 23D + right hand 12DOF (legacy)
        - 47D action: base 23D + left hand 12DOF + right hand 12DOF (TWIST style)
        """
        action_dim = int(actions.shape[-1])

        left_hand_state = actions[:, LEFT_HAND_STATE_IDX].squeeze(0).cpu()
        right_hand_state = actions[:, RIGHT_HAND_STATE_IDX].squeeze(0).cpu()

        if action_dim >= ACTION_DIM_WITH_BILATERAL_FINGERS:
            left_hand_state = (
                actions[:, LEFT_FINGER_ANGLES_START_IDX:LEFT_FINGER_ANGLES_END_IDX].squeeze(0).cpu().numpy()
            )
            right_hand_state = (
                actions[:, BILATERAL_RIGHT_FINGER_ANGLES_START_IDX:BILATERAL_RIGHT_FINGER_ANGLES_END_IDX]
                .squeeze(0)
                .cpu()
                .numpy()
            )
        elif action_dim >= ACTION_DIM_WITH_RIGHT_FINGERS:
            right_hand_state = (
                actions[:, RIGHT_FINGER_ANGLES_START_IDX:RIGHT_FINGER_ANGLES_END_IDX].squeeze(0).cpu().numpy()
            )
        else:
            assert action_dim >= BASE_ACTION_DIM, f"Invalid WBC-PINK action dim: {action_dim}"

        return left_hand_state, right_hand_state

    def _build_nav_straight_arm_targets(self) -> np.ndarray:
        """Build navigation-phase straight-arm targets in upper-body joint order."""
        q_default = np.asarray(self.robot_model.q_default, dtype=np.float64).reshape(-1)
        targets = q_default[self._upper_body_joint_indices].copy().astype(np.float64)

        upper_idx_by_name: dict[str, int] = {}
        for i, full_idx in enumerate(self._upper_body_joint_indices):
            joint_name = self._full_index_to_joint_name.get(int(full_idx), None)
            if joint_name is not None:
                upper_idx_by_name[joint_name] = int(i)

        def _env_float(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, str(default)))
            except ValueError:
                return float(default)

        # Straight hanging arm defaults. Can be tuned via env without code change.
        # NOTE:
        # - For G1, elbow ~= 1.2-1.35 rad is closer to a visual hanging arm than elbow=0.
        # - Right shoulder roll is mirrored by default (negative of left).
        shoulder_pitch = _env_float("G1_NAV_STRAIGHT_ARM_SHOULDER_PITCH", 0.0)
        shoulder_roll_left = _env_float("G1_NAV_STRAIGHT_ARM_SHOULDER_ROLL", 0.0)
        shoulder_roll_right = _env_float("G1_NAV_STRAIGHT_ARM_SHOULDER_ROLL_RIGHT", -shoulder_roll_left)
        shoulder_yaw = _env_float("G1_NAV_STRAIGHT_ARM_SHOULDER_YAW", 0.0)
        elbow = _env_float("G1_NAV_STRAIGHT_ARM_ELBOW", 1.25)
        wrist_roll = _env_float("G1_NAV_STRAIGHT_ARM_WRIST_ROLL", 0.0)
        wrist_pitch = _env_float("G1_NAV_STRAIGHT_ARM_WRIST_PITCH", 0.0)
        wrist_yaw = _env_float("G1_NAV_STRAIGHT_ARM_WRIST_YAW", 0.0)

        def _set_if_present(joint_name: str, value: float) -> None:
            idx = upper_idx_by_name.get(joint_name, None)
            if idx is not None:
                targets[idx] = float(value)

        _set_if_present("left_shoulder_pitch_joint", shoulder_pitch)
        _set_if_present("left_shoulder_roll_joint", shoulder_roll_left)
        _set_if_present("left_shoulder_yaw_joint", shoulder_yaw)
        _set_if_present("left_elbow_joint", elbow)
        _set_if_present("left_wrist_roll_joint", wrist_roll)
        _set_if_present("left_wrist_pitch_joint", wrist_pitch)
        _set_if_present("left_wrist_yaw_joint", wrist_yaw)

        _set_if_present("right_shoulder_pitch_joint", shoulder_pitch)
        _set_if_present("right_shoulder_roll_joint", shoulder_roll_right)
        _set_if_present("right_shoulder_yaw_joint", shoulder_yaw)
        _set_if_present("right_elbow_joint", elbow)
        _set_if_present("right_wrist_roll_joint", wrist_roll)
        _set_if_present("right_wrist_pitch_joint", wrist_pitch)
        _set_if_present("right_wrist_yaw_joint", wrist_yaw)

        return targets

    def _should_use_nav_straight_arm(self, navigate_cmd: torch.Tensor) -> bool:
        if not self._nav_straight_arm_enable:
            return False
        if self.cfg.use_p_control and self._is_navigating:
            return True
        if navigate_cmd.numel() == 0:
            return False
        nav_speed = float(torch.linalg.vector_norm(navigate_cmd, dim=-1).max().item())
        return nav_speed > self._nav_straight_arm_speed_threshold

    # """
    # Operations.
    # """
    def process_actions(self, actions: torch.Tensor):
        """Process the input actions and set targets for each task.

        Args:
            actions: The input actions tensor.

            action tensor layout:
            action = [left_hand_state: dim=1, 0 for open, 1 for close,
                      right_hand_state: dim=1, 0 for open, 1 for close,
                      left_arm_pos: dim=3, xyz position,
                      left_arm_quat: dim=4, wxyz quaternion,
                      right_arm_pos: dim=3, xyz position,
                      right_arm_quat: dim=4, wxyz quaternion,
                      navigate_cmd: dim=3, xyz velocity,
                      base_height_cmd: dim=1, height,
                      torso_orientation_rpy_cmd: dim=3, rpy]
        """

        # Store the raw actions
        self._raw_actions[:] = actions[:, : self.action_dim]

        # Make a copy of actions before modifying so that raw actions are not modified
        actions_clone = actions.clone()

        """
        **************************************************
        Upper body PINK controller
        **************************************************
        """
        # Extract upper body left/right arm pos/quat from actions
        left_arm_pos = actions_clone[:, LEFT_WRIST_POS_START_IDX:LEFT_WRIST_POS_END_IDX].squeeze(0).cpu()
        left_arm_quat = actions_clone[:, LEFT_WRIST_QUAT_START_IDX:LEFT_WRIST_QUAT_END_IDX].squeeze(0).cpu()
        right_arm_pos = actions_clone[:, RIGHT_WRIST_POS_START_IDX:RIGHT_WRIST_POS_END_IDX].squeeze(0).cpu()
        right_arm_quat = actions_clone[:, RIGHT_WRIST_QUAT_START_IDX:RIGHT_WRIST_QUAT_END_IDX].squeeze(0).cpu()

        # Convert from pos/quat to 4x4 transform matrix
        # Scipy requires quat xyzw, IsaacLab uses wxyz so a conversion is needed
        left_arm_quat = np.roll(left_arm_quat, -1)
        right_arm_quat = np.roll(right_arm_quat, -1)
        left_rotmat = R.from_quat(left_arm_quat).as_matrix()
        right_rotmat = R.from_quat(right_arm_quat).as_matrix()

        left_arm_pose = np.eye(4)
        left_arm_pose[:3, :3] = left_rotmat
        left_arm_pose[:3, 3] = left_arm_pos

        right_arm_pose = np.eye(4)
        right_arm_pose[:3, :3] = right_rotmat
        right_arm_pose[:3, 3] = right_arm_pos

        # Extract left/right hand state from actions.
        left_hand_state, right_hand_state = self._extract_hand_states_from_actions(actions_clone)

        # Assemble data format for running IK
        body_data = {LEFT_WRIST_LINK_NAME: left_arm_pose, RIGHT_WRIST_LINK_NAME: right_arm_pose}

        # Run IK
        target_robot_joints = self.compute_upperbody_joint_positions(body_data, left_hand_state, right_hand_state)

        # Reformat the joint position tensor to the correct order for G1 upper body
        target_upper_body_joints = target_robot_joints[self.robot_model.get_joint_group_indices("upper_body")]

        """
        **************************************************
        WBC closedloop
        **************************************************
        """
        # Extract navigate_cmd  base_height_cmd, and torso_orientation_rpy_cmd from actions
        navigate_cmd = self.get_navigation_cmd_from_actions(actions_clone)
        base_height_cmd = self.get_base_height_cmd_from_actions(actions_clone)
        torso_orientation_rpy_cmd = self.get_torso_orientation_rpy_cmd_from_actions(actions_clone)

        if self.cfg.use_p_control:
            if not self._is_navigating and self._navigation_goal_reached:
                self._navigation_goal_reached = False

            # Set flag for mimic to indicate that the robot has entered a navigation segment
            if not self._is_navigating and (np.abs(navigate_cmd) > NAVIGATE_THRESHOLD).any():
                self._is_navigating = True
                self._navigation_step_counter = 0
                self.navigation_p_controller.set_navigation_step_counter(self._navigation_step_counter)

            # Start applying navigation P-controller if conditions are met
            if self._is_navigating:
                assert self.cfg.navigation_subgoals is not None
                assert len(self.cfg.navigation_subgoals) > 0
                self._navigation_step_counter = self.navigation_p_controller.navigation_step_counter

                # No more subgoals to navigate to, stop navigation
                if (
                    self._num_navigation_subgoals_reached == len(self.cfg.navigation_subgoals) - 1
                ) or self._navigation_step_counter > self.cfg.max_navigation_steps:
                    computed_lin_vel_x, computed_lin_vel_y, computed_ang_vel = 0, 0, 0
                    self._is_navigating = False
                    self._navigation_goal_reached = True
                else:
                    target_xy_heading = self.cfg.navigation_subgoals[self._num_navigation_subgoals_reached + 1][0]
                    self.navigation_p_controller.set_inplace_turning_flag(
                        self.cfg.navigation_subgoals[self._num_navigation_subgoals_reached + 1][1]
                    )

                    target_xy = torch.tensor(target_xy_heading[:2])
                    target_heading = torch.tensor(target_xy_heading[2])
                    current_xy = self._asset.data.root_link_pos_w
                    current_heading = self._asset.data.heading_w

                    check_xy_reached = self.navigation_p_controller.check_xy_within_threshold(target_xy, current_xy)
                    check_heading_reached = self.navigation_p_controller.check_heading_within_threshold(
                        target_heading, current_heading
                    )

                    if check_xy_reached and check_heading_reached:
                        self._num_navigation_subgoals_reached += 1
                        computed_lin_vel_x, computed_lin_vel_y, computed_ang_vel = 0, 0, 0

                        self._is_navigating = False
                        self._navigation_goal_reached = True

                    # only turing in place, but may be deviated from the command xy position
                    elif check_heading_reached and self.navigation_p_controller.inplace_turning_flag:
                        computed_lin_vel_x, computed_lin_vel_y, computed_ang_vel = 0, 0, 0

                        self._num_navigation_subgoals_reached += 1
                        self._is_navigating = False
                        self._navigation_goal_reached = True

                    else:

                        computed_lin_vel_x, computed_lin_vel_y, computed_ang_vel = (
                            self.navigation_p_controller.run_p_controller(
                                target_heading=target_heading,
                                current_heading=current_heading,
                                target_xy=target_xy,
                                current_xy=current_xy,
                            )
                        )
                        # get single value out from the tensor
                        if isinstance(computed_lin_vel_x, torch.Tensor):
                            computed_lin_vel_x = computed_lin_vel_x.item()
                        if isinstance(computed_lin_vel_y, torch.Tensor):
                            computed_lin_vel_y = computed_lin_vel_y.item()
                        if isinstance(computed_ang_vel, torch.Tensor):
                            computed_ang_vel = computed_ang_vel.item()

                navigate_cmd[:, 0] = computed_lin_vel_x
                navigate_cmd[:, 1] = computed_lin_vel_y
                navigate_cmd[:, 2] = computed_ang_vel

        if self._should_use_nav_straight_arm(navigate_cmd):
            target_upper_body_joints = self._nav_straight_arm_targets.copy()

        self._navigate_cmd = navigate_cmd.clone()

        self.set_wbc_goal(navigate_cmd, base_height_cmd, torso_orientation_rpy_cmd)
        self.wbc_policy.set_goal(self._wbc_goal)

        """
        **************************************************
        Prepare WBC policy input
        **************************************************
        """
        wbc_obs = prepare_observations(self.num_envs, self._asset.data, self.wbc_g1_joints_order)
        self.wbc_policy.set_observation(wbc_obs)

        wbc_action = self.wbc_policy.get_action(target_upper_body_joints)
        self._processed_actions = postprocess_actions(
            wbc_action, self._asset.data, self.wbc_g1_joints_order, self.device
        )

    def apply_actions(self):
        """Apply the computed joint positions based on the WBC solution."""
        self._asset.set_joint_position_target(self._processed_actions, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term for specified environments.
        Args:
            env_ids: A list of environment IDs to reset. If None, all environments are reset.
        """
        self._raw_actions[env_ids] = torch.zeros(self.action_dim, device=self.device)

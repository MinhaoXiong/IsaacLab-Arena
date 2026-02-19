# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import pink
import pinocchio as pin
from pink.tasks import FrameTask, PostureTask

from isaaclab_arena_g1.g1_env.robot_model import ReducedRobotModel, RobotModel


class G1BodyIKSolverSettings:
    """
    G1 body IK solver settings.
    """

    def __init__(self):
        self.dt = 0.01
        self.num_step_per_frame = 1
        self.amplify_factor = 1.0
        self.posture_cost = 0.01
        self.posture_lm_damping = 1.0
        self.link_costs = {"hand": {"orientation_cost": 0.5, "position_cost": 1.0}}
        self.posture_weight = {
            "waist_pitch": 10.0,
            "shoulder_pitch": 4.0,
            "shoulder_roll": 3.0,
            "shoulder_yaw": 2.0,
        }


class WeightedPostureTask(PostureTask):
    """
    Weighted posture task for G1 body PINK IK solver.
    """

    def __init__(self, cost: float, weights: np.ndarray, lm_damping: float = 0.0, gain: float = 1.0) -> None:
        """Create weighted posture task.

        Args:
            cost: value used to cast joint angle differences to a homogeneous
                cost, in :math:`[\\mathrm{cost}] / [\\mathrm{rad}]`.
            weights: vector of weights for each joint.
            lm_damping: Unitless scale of the Levenberg-Marquardt (only when
                the error is large) regularization term, which helps when
                targets are unfeasible. Increase this value if the task is too
                jerky under unfeasible targets, but beware that too large a
                damping can slow down the task.
            gain: Task gain :math:`\alpha \\in [0, 1]` for additional low-pass
                filtering. Defaults to 1.0 (no filtering) for dead-beat
                control.
        """
        super().__init__(cost=cost, lm_damping=lm_damping, gain=gain)
        self.weights = weights

    def compute_error(self, configuration):
        error = super().compute_error(configuration)
        return self.weights * error

    def compute_jacobian(self, configuration):
        J = super().compute_jacobian(configuration)
        return self.weights[:, np.newaxis] * J

    def __repr__(self):
        """Human-readable representation of the weighted posture task."""
        return (
            "WeightedPostureTask("
            f"cost={self.cost}, "
            f"weights={self.weights}, "
            f"gain={self.gain}, "
            f"lm_damping={self.lm_damping})"
        )


class G1BodyIKSolver:
    """
    G1 body inverse kinematics solver.
    """

    def __init__(self, ik_solver_settings: G1BodyIKSolverSettings):
        self.dt = ik_solver_settings.dt
        self.num_step_per_frame = ik_solver_settings.num_step_per_frame
        self.amplify_factor = ik_solver_settings.amplify_factor
        self.link_costs = ik_solver_settings.link_costs
        self.posture_weight = ik_solver_settings.posture_weight
        self.posture_cost = ik_solver_settings.posture_cost
        self.posture_lm_damping = ik_solver_settings.posture_lm_damping
        self.robot = None

    def register_robot(self, robot):
        self.robot = robot
        self.initialize()

    def initialize(self):
        self.configuration = pink.Configuration(
            self.robot.pinocchio_wrapper.model,
            self.robot.pinocchio_wrapper.data,
            self.robot.q_default,
        )
        self.configuration.model.lowerPositionLimit = self.robot.lower_joint_limits
        self.configuration.model.upperPositionLimit = self.robot.upper_joint_limits

        # initialize tasks
        self.tasks = {}
        for link_name, weight in self.link_costs.items():
            assert link_name != "posture", "posture is a reserved task name"

            # Map robot-agnostic link names to robot-specific names
            if link_name == "hand":
                # Use hand_frame_names from supplemental info
                for side in ["left", "right"]:
                    frame_name = self.robot.supplemental_info.hand_frame_names[side]
                    task = FrameTask(
                        frame_name,
                        **weight,
                    )
                    self.tasks[frame_name] = task
            else:
                # For other links, use the name directly
                task = FrameTask(
                    link_name,
                    **weight,
                )
                self.tasks[link_name] = task

        # Add posture task
        if self.posture_weight is not None:
            weight = np.ones(self.robot.num_dofs)

            # Map robot-agnostic joint types to specific robot joint names using supplemental info
            for joint_type, posture_weight in self.posture_weight.items():
                if joint_type not in self.robot.supplemental_info.joint_name_mapping:
                    print(f"Warning: Unknown joint type {joint_type}")
                    continue

                # Get the joint name mapping for this type
                joint_mapping = self.robot.supplemental_info.joint_name_mapping[joint_type]

                # Handle both single joint names and left/right mappings
                if isinstance(joint_mapping, str):
                    # Single joint (e.g., waist joints)
                    if joint_mapping in self.robot.joint_to_dof_index:
                        joint_idx = self.robot.joint_to_dof_index[joint_mapping]
                        weight[joint_idx] = posture_weight
                else:
                    # Left/right mapping (e.g., arm joints)
                    for side in ["left", "right"]:
                        joint_name = joint_mapping[side]
                        if joint_name in self.robot.joint_to_dof_index:
                            joint_idx = self.robot.joint_to_dof_index[joint_name]
                            weight[joint_idx] = posture_weight

            self.tasks["posture"] = WeightedPostureTask(
                cost=self.posture_cost,
                weights=weight,
                lm_damping=self.posture_lm_damping,
            )
        else:
            self.tasks["posture"] = PostureTask(cost=self.posture_cost, lm_damping=self.posture_lm_damping)
        for task in self.tasks.values():
            task.set_target_from_configuration(self.configuration)

    def __call__(self, target_pose: dict):
        for link_name, pose in target_pose.items():
            if link_name not in self.tasks:
                continue
            pose = pin.SE3(pose[:3, :3], pose[:3, 3])
            self.tasks[link_name].set_target(pose)

        for _ in range(self.num_step_per_frame):
            velocity = pink.solve_ik(
                self.configuration,
                self.tasks.values(),
                dt=self.dt,
                solver="osqp",
            )
            self.configuration.q = self.robot.clip_configuration(
                self.configuration.q + velocity * self.dt * self.amplify_factor
            )
            self.configuration.update()
            self.robot.cache_forward_kinematics(self.configuration.q)

        return self.configuration.q.copy()

    def update_weights(self, weights):
        for link_name, weight in weights.items():
            if "position_cost" in weight:
                self.tasks[link_name].set_position_cost(weight["position_cost"])
            if "orientation_cost" in weight:
                self.tasks[link_name].set_orientation_cost(weight["orientation_cost"])


class G1WBCUpperbodyController:
    """
    G1 PINK upper body controller for GR00T WBC.
    """

    def __init__(
        self,
        robot_model: RobotModel,
        body_active_joint_groups: list[str] | None = None,
    ):
        # Initialize the body
        if body_active_joint_groups is not None:
            self.body = ReducedRobotModel.from_active_groups(robot_model, body_active_joint_groups)
            self.full_robot = self.body.full_robot
            self.using_reduced_robot_model = True
        else:
            self.body = robot_model
            self.full_robot = self.body
            self.using_reduced_robot_model = False
        body_ik_solver_settings = G1BodyIKSolverSettings()
        self.body_ik_solver = G1BodyIKSolver(body_ik_solver_settings)

        # Register the specific robot model to the robot-agnostic body IK solver class
        self.body_ik_solver.register_robot(self.body)

        self.in_warmup = True

    # BODex joint order -> URDF joint order mapping (12 DOF)
    # BODex: [thumb_yaw, thumb_pitch, index, middle, ring, pinky,
    #         thumb_inter, thumb_distal, index_inter, middle_inter, ring_inter, pinky_inter]
    # URDF:  [index_prox, index_inter, middle_prox, middle_inter, pinky_prox, pinky_inter,
    #         ring_prox, ring_inter, thumb_yaw, thumb_pitch, thumb_inter, thumb_distal]
    _BODEX_TO_URDF = [2, 8, 3, 9, 5, 11, 4, 10, 0, 1, 6, 7]

    def get_hand_joint_pos(self, hand_state):
        """Map hand state to 12 InspireHand joint angles in URDF order.

        hand_state: scalar 0=open, 1=close, or np.ndarray of 12 joint angles (BODex order).
        """
        if isinstance(hand_state, np.ndarray) and hand_state.size == 12:
            # Reorder from BODex order to URDF order
            return hand_state[self._BODEX_TO_URDF].copy()

        # URDF order: index, index_i, middle, middle_i, pinky, pinky_i,
        #             ring, ring_i, thumb_yaw, thumb_pitch, thumb_i, thumb_d
        hand_q = np.zeros(12)
        if hand_state == 0:
            return hand_q

        # Closed pose
        index_v = 1.2
        thumb_pitch_v = 0.35
        hand_q[0] = index_v   # index_prox
        hand_q[1] = index_v   # index_inter (mimic 1.0x)
        hand_q[2] = index_v   # middle_prox
        hand_q[3] = index_v   # middle_inter
        hand_q[4] = index_v   # pinky_prox
        hand_q[5] = index_v   # pinky_inter
        hand_q[6] = index_v   # ring_prox
        hand_q[7] = index_v   # ring_inter
        hand_q[8] = 0.9       # thumb_yaw
        hand_q[9] = thumb_pitch_v  # thumb_pitch
        hand_q[10] = thumb_pitch_v * 1.6  # thumb_inter
        hand_q[11] = thumb_pitch_v * 2.4  # thumb_distal
        return hand_q

    def inverse_kinematics(
        self,
        body_target_pose,
        left_hand_state,
        right_hand_state,
    ):
        """
        Solve the inverse kinematics problem for the given target poses.
        Args:
            body_target_pose: Dictionary of link names and their corresponding target pose.
            left_hand_target_pose: Binary gripper state (0 for open, 1 for close).
            right_hand_target_pose: Binary gripper state (0 for open, 1 for close).
        Returns:
            G1 joint position vector that achieves the target poses.
        """

        if self.using_reduced_robot_model:
            body_q = self.body.reduced_to_full_configuration(self.body_ik_solver(body_target_pose))
        else:
            body_q = self.body_ik_solver(body_target_pose)

        left_hand_joint_pos = self.get_hand_joint_pos(left_hand_state)
        right_hand_joint_pos = self.get_hand_joint_pos(right_hand_state)

        body_q[self.full_robot.get_hand_actuated_joint_indices(side="left")] = left_hand_joint_pos
        body_q[self.full_robot.get_hand_actuated_joint_indices(side="right")] = right_hand_joint_pos

        return body_q

# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationData

_LEGACY_HAND_TO_INSPIRE_HAND = {
    "left_hand_index_0_joint": ["L_index_proximal_joint"],
    "left_hand_index_1_joint": ["L_index_intermediate_joint"],
    "left_hand_middle_0_joint": ["L_middle_proximal_joint", "L_ring_proximal_joint", "L_pinky_proximal_joint"],
    "left_hand_middle_1_joint": [
        "L_middle_intermediate_joint",
        "L_ring_intermediate_joint",
        "L_pinky_intermediate_joint",
    ],
    "left_hand_ring_0_joint": ["L_ring_proximal_joint"],
    "left_hand_ring_1_joint": ["L_ring_intermediate_joint"],
    "left_hand_pinky_0_joint": ["L_pinky_proximal_joint"],
    "left_hand_pinky_1_joint": ["L_pinky_intermediate_joint"],
    "left_hand_little_0_joint": ["L_pinky_proximal_joint"],
    "left_hand_little_1_joint": ["L_pinky_intermediate_joint"],
    "left_hand_thumb_0_joint": ["L_thumb_proximal_yaw_joint"],
    "left_hand_thumb_1_joint": ["L_thumb_proximal_pitch_joint"],
    "left_hand_thumb_2_joint": ["L_thumb_intermediate_joint", "L_thumb_distal_joint"],
    "right_hand_index_0_joint": ["R_index_proximal_joint"],
    "right_hand_index_1_joint": ["R_index_intermediate_joint"],
    "right_hand_middle_0_joint": ["R_middle_proximal_joint", "R_ring_proximal_joint", "R_pinky_proximal_joint"],
    "right_hand_middle_1_joint": [
        "R_middle_intermediate_joint",
        "R_ring_intermediate_joint",
        "R_pinky_intermediate_joint",
    ],
    "right_hand_ring_0_joint": ["R_ring_proximal_joint"],
    "right_hand_ring_1_joint": ["R_ring_intermediate_joint"],
    "right_hand_pinky_0_joint": ["R_pinky_proximal_joint"],
    "right_hand_pinky_1_joint": ["R_pinky_intermediate_joint"],
    "right_hand_little_0_joint": ["R_pinky_proximal_joint"],
    "right_hand_little_1_joint": ["R_pinky_intermediate_joint"],
    "right_hand_thumb_0_joint": ["R_thumb_proximal_yaw_joint"],
    "right_hand_thumb_1_joint": ["R_thumb_proximal_pitch_joint"],
    "right_hand_thumb_2_joint": ["R_thumb_intermediate_joint", "R_thumb_distal_joint"],
}

_INSPIRE_HAND_TO_LEGACY_HAND_CANDIDATES = {
    "L_index_proximal_joint": ["left_hand_index_0_joint"],
    "L_index_intermediate_joint": ["left_hand_index_1_joint"],
    "L_middle_proximal_joint": ["left_hand_middle_0_joint"],
    "L_middle_intermediate_joint": ["left_hand_middle_1_joint"],
    "L_ring_proximal_joint": ["left_hand_ring_0_joint", "left_hand_middle_0_joint"],
    "L_ring_intermediate_joint": ["left_hand_ring_1_joint", "left_hand_middle_1_joint"],
    "L_pinky_proximal_joint": ["left_hand_pinky_0_joint", "left_hand_little_0_joint", "left_hand_middle_0_joint"],
    "L_pinky_intermediate_joint": [
        "left_hand_pinky_1_joint",
        "left_hand_little_1_joint",
        "left_hand_middle_1_joint",
    ],
    "L_thumb_proximal_yaw_joint": ["left_hand_thumb_0_joint"],
    "L_thumb_proximal_pitch_joint": ["left_hand_thumb_1_joint"],
    "L_thumb_intermediate_joint": ["left_hand_thumb_2_joint"],
    "L_thumb_distal_joint": ["left_hand_thumb_2_joint"],
    "R_index_proximal_joint": ["right_hand_index_0_joint"],
    "R_index_intermediate_joint": ["right_hand_index_1_joint"],
    "R_middle_proximal_joint": ["right_hand_middle_0_joint"],
    "R_middle_intermediate_joint": ["right_hand_middle_1_joint"],
    "R_ring_proximal_joint": ["right_hand_ring_0_joint", "right_hand_middle_0_joint"],
    "R_ring_intermediate_joint": ["right_hand_ring_1_joint", "right_hand_middle_1_joint"],
    "R_pinky_proximal_joint": [
        "right_hand_pinky_0_joint",
        "right_hand_little_0_joint",
        "right_hand_middle_0_joint",
    ],
    "R_pinky_intermediate_joint": [
        "right_hand_pinky_1_joint",
        "right_hand_little_1_joint",
        "right_hand_middle_1_joint",
    ],
    "R_thumb_proximal_yaw_joint": ["right_hand_thumb_0_joint"],
    "R_thumb_proximal_pitch_joint": ["right_hand_thumb_1_joint"],
    "R_thumb_intermediate_joint": ["right_hand_thumb_2_joint"],
    "R_thumb_distal_joint": ["right_hand_thumb_2_joint"],
}

_WARNED_KEYS = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED_KEYS:
        print(message)
        _WARNED_KEYS.add(key)


def _resolve_sim_joint_to_wbc_joint_names(sim_joint_name: str, wbc_joints_order: dict[str, int]) -> list[str]:
    if sim_joint_name in wbc_joints_order:
        return [sim_joint_name]
    if sim_joint_name in _LEGACY_HAND_TO_INSPIRE_HAND:
        return [name for name in _LEGACY_HAND_TO_INSPIRE_HAND[sim_joint_name] if name in wbc_joints_order]
    return []


def _resolve_wbc_joint_to_sim_joint_name(wbc_joint_name: str, sim_joint_name_to_index: dict[str, int]) -> str:
    if wbc_joint_name in sim_joint_name_to_index:
        return wbc_joint_name
    for candidate in _INSPIRE_HAND_TO_LEGACY_HAND_CANDIDATES.get(wbc_joint_name, []):
        if candidate in sim_joint_name_to_index:
            return candidate
    return ""


def convert_sim_joint_to_wbc_joint(
    sim_joint_data: np.ndarray, sim_joint_names: list[str], wbc_joints_order: dict[str, int]
) -> np.ndarray:
    """Convert sim joint observations to WBC joint observations.

    Args:
        sim_joint_data: Sim joint data in Lab's order
        sim_joint_names: Sim joint names in Lab's order
        wbc_joints_order: WBC joint order in policy config yaml

    Returns:
        WBC joint data in WBC joint order
    """
    # Check if sim_joint_data is a numpy array, if not, convert from torch tensor to numpy
    if not isinstance(sim_joint_data, np.ndarray):
        sim_joint_data = sim_joint_data.cpu().numpy()

    num_joints = len(wbc_joints_order)
    num_envs = sim_joint_data.shape[0]
    wbc_joint_data = np.zeros((num_envs, num_joints), dtype=sim_joint_data.dtype)
    wbc_joint_filled = np.zeros((num_joints,), dtype=bool)
    missing_joint_names = []

    for sim_joint_index, sim_joint_name in enumerate(sim_joint_names):
        mapped_wbc_joint_names = _resolve_sim_joint_to_wbc_joint_names(sim_joint_name, wbc_joints_order)
        if not mapped_wbc_joint_names:
            missing_joint_names.append(sim_joint_name)
            continue
        for wbc_joint_name in mapped_wbc_joint_names:
            wbc_joint_index = wbc_joints_order[wbc_joint_name]
            wbc_joint_data[:, wbc_joint_index] = sim_joint_data[:, sim_joint_index]
            wbc_joint_filled[wbc_joint_index] = True

    if missing_joint_names:
        raise AssertionError(
            f"Unmapped sim joints in convert_sim_joint_to_wbc_joint: {missing_joint_names}. "
            "Please update legacy<->Inspire joint aliases."
        )

    if not np.all(wbc_joint_filled):
        wbc_joint_name_by_index = {joint_idx: joint_name for joint_name, joint_idx in wbc_joints_order.items()}
        missing_wbc_joint_names = [
            wbc_joint_name_by_index[joint_idx] for joint_idx in range(num_joints) if not wbc_joint_filled[joint_idx]
        ]
        _warn_once(
            "missing_wbc_obs_joints",
            f"[run_policy] Missing WBC joints in observation map (filled with zeros): {missing_wbc_joint_names}",
        )

    return wbc_joint_data


def prepare_observations(
    num_envs: int, robot_data: ArticulationData, wbc_joints_order: dict[str, int]
) -> dict[str, np.ndarray]:
    """Prepare observations for the policy.

    Args:
        num_envs: Number of environments
        robot_data: Robot data
        wbc_joints_order: WBC joint order in policy config yaml

    Returns:
        Observations for the policy
        - q: Joint positions
        - dq: Joint velocities
        - ddq: Joint accelerations
        - floating_base_pose: Floating base pose
        - floating_base_vel: Floating base velocity
        - floating_base_acc: Floating base acceleration
        - torso_quat: Torso quaternion
        - torso_ang_vel: Torso angular velocity
    """
    # Get robot joint observations
    sim_joint_pos = robot_data.joint_pos.cpu().numpy()
    sim_joint_vel = robot_data.joint_vel.cpu().numpy()
    wbc_num_joints = len(wbc_joints_order)

    # Convert joints data from Lab's order to GR00T's order saved in config yaml
    wbc_joint_pos = np.zeros((num_envs, wbc_num_joints))
    wbc_joint_vel = np.zeros((num_envs, wbc_num_joints))
    wbc_joint_acc = np.zeros((num_envs, wbc_num_joints))
    wbc_joint_pos = convert_sim_joint_to_wbc_joint(sim_joint_pos, robot_data.joint_names, wbc_joints_order)
    wbc_joint_vel = convert_sim_joint_to_wbc_joint(sim_joint_vel, robot_data.joint_names, wbc_joints_order)

    # Prepare obs dict for WBC policy input to G1DecoupledWholeBodyPolicy class
    assert wbc_joint_pos.shape == wbc_joint_vel.shape == wbc_joint_acc.shape == (num_envs, wbc_num_joints)

    root_link_pos_w = robot_data.root_link_pos_w.cpu().numpy()
    root_link_quat_w = robot_data.root_link_quat_w.cpu().numpy()
    base_pose_w = np.concatenate((root_link_pos_w, root_link_quat_w), axis=1)
    base_lin_vel_b = robot_data.root_link_lin_vel_b.cpu().numpy()
    base_ang_vel_b = robot_data.root_link_ang_vel_b.cpu().numpy()

    base_vel_b = np.concatenate((base_lin_vel_b, base_ang_vel_b), axis=1)
    # torso link in world frame
    torso_link_pose_w = robot_data.body_link_state_w[:, robot_data.body_names.index("torso_link"), :]
    torso_link_quat_w = torso_link_pose_w[:, 3:7]  # w, x, y, z
    torso_link_ang_vel_w = torso_link_pose_w[:, -3:]

    torso_link_ang_vel_b = math_utils.quat_apply_inverse(torso_link_quat_w, torso_link_ang_vel_w)

    # Prepare obs tmers
    wbc_obs = {
        "q": wbc_joint_pos,
        "dq": wbc_joint_vel,
        "ddq": np.zeros((num_envs, wbc_num_joints)),  # Not used by Standing Waist Height Policy
        "tau_est": np.zeros((num_envs, wbc_num_joints)),  # Not used by Standing Waist Height Policy
        "floating_base_pose": base_pose_w,  # wrt world frame, used to project gravity vector to local frame
        "floating_base_vel": base_vel_b,  # wrt body frame
        "floating_base_acc": np.zeros((num_envs, 6)),  # Not used by Standing Waist Height Policy
        "torso_quat": torso_link_quat_w.cpu().numpy(),
        "torso_ang_vel": torso_link_ang_vel_b.cpu().numpy(),
    }
    return wbc_obs


def postprocess_actions(
    wbc_action: dict[str, np.ndarray],
    robot_data: ArticulationData,
    wbc_g1_joints_order: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    """Postprocess actions for the policy."""
    num_envs = wbc_action["q"].shape[0]
    num_joints = len(robot_data.joint_names)
    processed_actions = torch.zeros((num_envs, num_joints), device=device)
    wbc_joints_pos_action = torch.from_numpy(wbc_action["q"]).to(device=device, dtype=processed_actions.dtype)
    sim_joint_name_to_index = {joint_name: joint_index for joint_index, joint_name in enumerate(robot_data.joint_names)}

    # Convert WBC joints order to Lab joints order (with Inspire<->legacy hand aliases when needed).
    accum = torch.zeros_like(processed_actions)
    accum_count = torch.zeros((num_joints,), device=device, dtype=processed_actions.dtype)
    missing_wbc_joint_names = []
    out_of_range_wbc_joint_names = []
    policy_action_dim = int(wbc_joints_pos_action.shape[1])

    for wbc_joint_name, wbc_joint_index in wbc_g1_joints_order.items():
        if int(wbc_joint_index) >= policy_action_dim:
            out_of_range_wbc_joint_names.append(wbc_joint_name)
            continue
        sim_joint_name = _resolve_wbc_joint_to_sim_joint_name(wbc_joint_name, sim_joint_name_to_index)
        if sim_joint_name not in sim_joint_name_to_index:
            missing_wbc_joint_names.append(wbc_joint_name)
            continue
        sim_joint_index = sim_joint_name_to_index[sim_joint_name]
        accum[:, sim_joint_index] += wbc_joints_pos_action[:, wbc_joint_index]
        accum_count[sim_joint_index] += 1.0

    if missing_wbc_joint_names:
        _warn_once(
            "missing_wbc_action_joints",
            f"[run_policy] Skipped unmapped WBC action joints: {missing_wbc_joint_names}",
        )
    if out_of_range_wbc_joint_names:
        _warn_once(
            "out_of_range_wbc_action_joints",
            "[run_policy] Skipped WBC joints with index >= policy action dim "
            f"({policy_action_dim}): {out_of_range_wbc_joint_names}",
        )

    valid_mask = accum_count > 0
    if torch.any(valid_mask):
        processed_actions[:, valid_mask] = accum[:, valid_mask] / accum_count[valid_mask]

    return processed_actions

# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Literal

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

from isaaclab_arena_g1.g1_env.g1_supplemental_info import (
    G1SupplementalInfo,
    G1SupplementalInfoWaistLowerAndUpperBody,
    G1SupplementalInfoWaistUpperBody,
)
from isaaclab_arena_g1.g1_env.robot_model import RobotModel

# Local G1+InspireHand URDF directory.
# Set G1_INSPIRE_HAND_URDF_DIR env var, or it defaults to
# <Humanoid-gen-pack>/configs/g1_inspirehand (detected via PACK_ROOT or relative to IsaacLab-Arena).
def _default_inspire_urdf_dir() -> str:
    env = os.environ.get("G1_INSPIRE_HAND_URDF_DIR")
    if env:
        return env
    pack_root = os.environ.get("PACK_ROOT")
    if pack_root:
        return os.path.join(pack_root, "configs", "g1_inspirehand")
    # Fallback: assume IsaacLab-Arena is under Humanoid-gen-pack/repos/
    return os.path.join(os.path.dirname(__file__), "../../../../../../configs/g1_inspirehand")

_INSPIRE_HAND_URDF_DIR = _default_inspire_urdf_dir()


def _parse_g1_hand_type_env(name: str, default: str = "inspire") -> str:
    """Parse G1 hand type selector from env var."""
    raw = os.environ.get(name, default).strip().lower()
    if raw in {"inspire", "inspirehand", "inspire_hand"}:
        return "inspire"
    if raw in {"dex3", "dex3-1", "dex", "unitree_dex3", "default"}:
        return "dex3"
    print(f"[g1] Warning: invalid {name}={raw!r}, fallback to {default!r}.")
    return default


def _resolve_default_wbc_urdf():
    robot_model_config = {
        "asset_path": f"{ISAACLAB_NUCLEUS_DIR}/Arena/wbc_policy/robot_model/g1/",
        "urdf_path": f"{ISAACLAB_NUCLEUS_DIR}/Arena/wbc_policy/robot_model/g1/g1_29dof_with_hand.urdf",
    }
    asset_path_local = retrieve_file_path(robot_model_config["asset_path"], force_download=True)
    urdf_path_local = retrieve_file_path(robot_model_config["urdf_path"], force_download=True)
    return asset_path_local, urdf_path_local


_G1_HAND_TYPE = _parse_g1_hand_type_env("G1_HAND_TYPE", default="inspire")


def instantiate_g1_robot_model(
    waist_location: Literal["lower_body", "upper_body"] = "lower_body",
):
    """
    Instantiate a G1 robot model with configurable waist location, and summarize the supplemental info.

    Args:
        waist_location: Whether to put waist in "lower_body" (default G1 behavior),
                        "upper_body" (waist controlled with arms/manipulation via IK),
                        or "lower_and_upper_body" (waist reference from arms/manipulation
                        via IK then passed to lower body policy)

    Returns:
        RobotModel: Configured G1 robot model
    """

    if _G1_HAND_TYPE == "inspire":
        # Use local G1+InspireHand URDF if available, otherwise fall back to default WBC URDF.
        local_urdf = os.path.join(_INSPIRE_HAND_URDF_DIR, "g1_29dof_with_inspire_hand.urdf")
        if os.path.isfile(local_urdf):
            urdf_path_local = os.path.abspath(local_urdf)
            asset_path_local = os.path.dirname(urdf_path_local)
            print(f"[g1] hand_type=inspire, local URDF: {urdf_path_local}")
        else:
            asset_path_local, urdf_path_local = _resolve_default_wbc_urdf()
            print(f"[g1] hand_type=inspire, fallback Nucleus URDF: {urdf_path_local}")
    else:
        asset_path_local, urdf_path_local = _resolve_default_wbc_urdf()
        print(f"[g1] hand_type=dex3, Nucleus URDF: {urdf_path_local}")

    assert waist_location in [
        "lower_body",
        "upper_body",
        "lower_and_upper_body",
    ], f"Invalid waist_location: {waist_location}. Must be 'lower_body' or 'upper_body' or 'lower_and_upper_body'"
    # Choose supplemental info based on waist location preference
    if waist_location == "lower_body":
        robot_model_supplemental_info = G1SupplementalInfo()
    elif waist_location == "upper_body":
        robot_model_supplemental_info = G1SupplementalInfoWaistUpperBody()
    elif waist_location == "lower_and_upper_body":
        robot_model_supplemental_info = G1SupplementalInfoWaistLowerAndUpperBody()

    robot_model = RobotModel(
        urdf_path_local,
        asset_path_local,
        supplemental_info=robot_model_supplemental_info,
    )
    return robot_model

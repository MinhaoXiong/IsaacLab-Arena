# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
import tempfile
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


_WBC_G1_NUCLEUS_ROOT = f"{ISAACLAB_NUCLEUS_DIR}/Arena/wbc_policy/robot_model/g1"
_WBC_G1_NUCLEUS_URDF = f"{_WBC_G1_NUCLEUS_ROOT}/g1_29dof_with_hand.urdf"
_MESH_FILENAME_PATTERN = re.compile(r'filename="([^"]+)"')


def _is_uri_or_abs_path(path: str) -> bool:
    path = path.strip()
    return (
        os.path.isabs(path)
        or "://" in path
        or path.startswith("package://")
        or path.startswith("file://")
    )


def _resolve_default_wbc_urdf():
    """Download dex3 URDF from Nucleus and localize all relative mesh paths."""
    cache_root = os.path.join(tempfile.gettempdir(), "isaaclab_arena_g1_dex3")
    os.makedirs(cache_root, exist_ok=True)

    urdf_path_local = retrieve_file_path(_WBC_G1_NUCLEUS_URDF, download_dir=cache_root, force_download=True)
    with open(urdf_path_local, encoding="utf-8") as f:
        urdf_text = f.read()

    localized_meshes: dict[str, str] = {}
    for mesh_name in set(_MESH_FILENAME_PATTERN.findall(urdf_text)):
        if _is_uri_or_abs_path(mesh_name):
            continue
        rel_mesh = mesh_name.lstrip("./")
        mesh_nucleus_path = f"{_WBC_G1_NUCLEUS_ROOT}/{rel_mesh}"
        mesh_download_dir = os.path.join(cache_root, os.path.dirname(rel_mesh))
        os.makedirs(mesh_download_dir, exist_ok=True)
        mesh_local_path = retrieve_file_path(mesh_nucleus_path, download_dir=mesh_download_dir, force_download=False)
        localized_meshes[mesh_name] = os.path.abspath(mesh_local_path).replace("\\", "/")

    def _replace_mesh_filename(match: re.Match[str]) -> str:
        original = match.group(1)
        localized = localized_meshes.get(original)
        if localized is None:
            return match.group(0)
        return f'filename="{localized}"'

    localized_urdf_path = os.path.join(cache_root, "g1_29dof_with_hand.localized.urdf")
    localized_urdf_text = _MESH_FILENAME_PATTERN.sub(_replace_mesh_filename, urdf_text)
    with open(localized_urdf_path, "w", encoding="utf-8") as f:
        f.write(localized_urdf_text)
    return cache_root, localized_urdf_path


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

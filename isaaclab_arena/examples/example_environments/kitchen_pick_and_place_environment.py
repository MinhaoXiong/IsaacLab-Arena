# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import math

from isaaclab_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase

# NOTE(alexmillane, 2025.09.04): There is an issue with type annotation in this file.
# We cannot annotate types which require the simulation app to be started in order to
# import, because this file is used to retrieve CLI arguments, so it must be imported
# before the simulation app is started.
# TODO(alexmillane, 2025.09.04): Fix this.


class KitchenPickAndPlaceEnvironment(ExampleEnvironmentBase):

    name: str = "kitchen_pick_and_place"

    @staticmethod
    def _parse_xyz(text: str, name: str) -> tuple[float, float, float]:
        values = [float(x.strip()) for x in text.split(",")]
        if len(values) != 3:
            raise ValueError(f"{name} must be csv xyz with 3 values, got: {text}")
        return (values[0], values[1], values[2])

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("kitchen")()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        pick_up_object.set_initial_pose(
            Pose(
                position_xyz=(0.4, 0.0, 0.1),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )
        # Keep G1 at a deterministic table-side spawn for arm-follow debugging.
        if args_cli.embodiment.startswith("g1_"):
            g1_init_pos_w = self._parse_xyz(args_cli.g1_init_pos_w, "g1_init_pos_w")
            yaw_rad = math.radians(float(args_cli.g1_init_yaw_deg))
            g1_init_quat_wxyz = (math.cos(0.5 * yaw_rad), 0.0, 0.0, math.sin(0.5 * yaw_rad))
            embodiment.set_initial_pose(Pose(position_xyz=g1_init_pos_w, rotation_wxyz=g1_init_quat_wxyz))

        # TODO(alexmillane, 2025.09.24): Add automatic object type detection of ObjectReferences.
        destination_location = ObjectReference(
            name="destination_location",
            prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )

        scene = Scene(assets=[background, pick_up_object, destination_location])
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=PickAndPlaceTask(pick_up_object, destination_location, background),
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default="cracker_box")
        parser.add_argument("--embodiment", type=str, default="franka")
        # NOTE(alexmillane, 2025.09.04): We need a teleop device argument in order
        # to be used in the record_demos.py script.
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument(
            "--g1-init-pos-w",
            type=str,
            default="0.05,0.0,0.0",
            help="G1 initial base position xyz in world frame for kitchen scene.",
        )
        parser.add_argument(
            "--g1-init-yaw-deg",
            type=float,
            default=0.0,
            help="G1 initial yaw in degrees for kitchen scene.",
        )

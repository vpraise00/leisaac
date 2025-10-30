# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waypoint controller for quad-arm manipulation."""

import torch
import json
from dataclasses import dataclass
from typing import Optional

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat, quat_inv


# Constants
ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
IDENTITY_QUAT = [1.0, 0.0, 0.0, 0.0]


@dataclass
class WaypointCommand:
    """Single waypoint command for quad-arm."""
    north_world_pose: torch.Tensor  # (1, 7) - position + quaternion
    east_world_pose: torch.Tensor  # (1, 7)
    west_world_pose: torch.Tensor  # (1, 7)
    south_world_pose: torch.Tensor  # (1, 7)
    north_gripper: float
    east_gripper: float
    west_gripper: float
    south_gripper: float
    hold_steps: int
    north_wrist_flex: Optional[float] = None
    east_wrist_flex: Optional[float] = None
    west_wrist_flex: Optional[float] = None
    south_wrist_flex: Optional[float] = None


class QuadArmWaypointController:
    """Controller for executing waypoint sequences on quad-arm robot."""

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        command_type: str = "position",
        ik_method: str = "dls",
        position_tol: float = 0.01,
        orientation_tol: float = 0.02,
        pose_interp_gain: float = 1.0,
        interp_gain: float = 1.0,
        force_wrist_down: bool = False,
        wrist_flex_angle: float = 1.57,
        wrist_flex_min: Optional[float] = None,
        wrist_flex_max: Optional[float] = None,
    ):
        """Initialize quad-arm waypoint controller."""
        self.env = env
        self.position_tol = position_tol
        self.orientation_tol = orientation_tol
        self.pose_interp_gain = pose_interp_gain
        self.interp_gain = interp_gain
        self.force_wrist_down = force_wrist_down
        self.wrist_flex_angle = wrist_flex_angle
        self.wrist_flex_min = wrist_flex_min
        self.wrist_flex_max = wrist_flex_max

        # Setup arms
        self.north_arm = env.scene["north_arm"]
        self.east_arm = env.scene["east_arm"]
        self.west_arm = env.scene["west_arm"]
        self.south_arm = env.scene["south_arm"]

        self.north_entity = SceneEntityCfg("north_arm", joint_names=ARM_JOINT_NAMES, body_names=["jaw"])
        self.east_entity = SceneEntityCfg("east_arm", joint_names=ARM_JOINT_NAMES, body_names=["jaw"])
        self.west_entity = SceneEntityCfg("west_arm", joint_names=ARM_JOINT_NAMES, body_names=["jaw"])
        self.south_entity = SceneEntityCfg("south_arm", joint_names=ARM_JOINT_NAMES, body_names=["jaw"])

        self.north_entity.resolve(env.scene)
        self.east_entity.resolve(env.scene)
        self.west_entity.resolve(env.scene)
        self.south_entity.resolve(env.scene)

        # IK Verification: Print end effector configuration
        print("\n" + "="*80)
        print("[IK VERIFICATION] End Effector Configuration Check")
        print("="*80)
        print(f"North arm:")
        print(f"  Body IDs: {self.north_entity.body_ids}")
        print(f"  Body names (from prim): {self.north_arm.body_names}")
        print(f"  Target body index: {self.north_entity.body_ids[0]}")
        print(f"  Joint IDs: {self.north_entity.joint_ids}")

        print(f"\nEast arm:")
        print(f"  Body IDs: {self.east_entity.body_ids}")
        print(f"  Body names (from prim): {self.east_arm.body_names}")
        print(f"  Target body index: {self.east_entity.body_ids[0]}")

        print(f"\nWest arm:")
        print(f"  Body IDs: {self.west_entity.body_ids}")
        print(f"  Body names (from prim): {self.west_arm.body_names}")
        print(f"  Target body index: {self.west_entity.body_ids[0]}")

        print(f"\nSouth arm:")
        print(f"  Body IDs: {self.south_entity.body_ids}")
        print(f"  Body names (from prim): {self.south_arm.body_names}")
        print(f"  Target body index: {self.south_entity.body_ids[0]}")

        # Check initial EE positions
        ee_pose_north = self.north_arm.data.body_pose_w[:, self.north_entity.body_ids[0]]
        ee_pose_east = self.east_arm.data.body_pose_w[:, self.east_entity.body_ids[0]]
        ee_pose_west = self.west_arm.data.body_pose_w[:, self.west_entity.body_ids[0]]
        ee_pose_south = self.south_arm.data.body_pose_w[:, self.south_entity.body_ids[0]]

        print(f"\nInitial End Effector Positions (World Frame):")
        print(f"  North EE: ({ee_pose_north[0, 0]:.4f}, {ee_pose_north[0, 1]:.4f}, {ee_pose_north[0, 2]:.4f})")
        print(f"  East  EE: ({ee_pose_east[0, 0]:.4f}, {ee_pose_east[0, 1]:.4f}, {ee_pose_east[0, 2]:.4f})")
        print(f"  West  EE: ({ee_pose_west[0, 0]:.4f}, {ee_pose_west[0, 1]:.4f}, {ee_pose_west[0, 2]:.4f})")
        print(f"  South EE: ({ee_pose_south[0, 0]:.4f}, {ee_pose_south[0, 1]:.4f}, {ee_pose_south[0, 2]:.4f})")
        print("="*80 + "\n")

        # Setup IK controllers
        ik_cfg = DifferentialIKControllerCfg(command_type=command_type, use_relative_mode=False, ik_method=ik_method)
        self.north_ik = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)
        self.east_ik = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)
        self.west_ik = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)
        self.south_ik = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)
        self.controls_orientation = ik_cfg.command_type == "pose"

        # Setup Jacobian indices
        self.north_jacobian_index = self.north_entity.body_ids[0] - 1 if self.north_arm.is_fixed_base else self.north_entity.body_ids[0]
        self.east_jacobian_index = self.east_entity.body_ids[0] - 1 if self.east_arm.is_fixed_base else self.east_entity.body_ids[0]
        self.west_jacobian_index = self.west_entity.body_ids[0] - 1 if self.west_arm.is_fixed_base else self.west_entity.body_ids[0]
        self.south_jacobian_index = self.south_entity.body_ids[0] - 1 if self.south_arm.is_fixed_base else self.south_entity.body_ids[0]

        # IK Verification: Print Jacobian indices
        print(f"[IK VERIFICATION] Jacobian Configuration:")
        print(f"  North - is_fixed_base: {self.north_arm.is_fixed_base}, jacobian_index: {self.north_jacobian_index}")
        print(f"  East  - is_fixed_base: {self.east_arm.is_fixed_base}, jacobian_index: {self.east_jacobian_index}")
        print(f"  West  - is_fixed_base: {self.west_arm.is_fixed_base}, jacobian_index: {self.west_jacobian_index}")
        print(f"  South - is_fixed_base: {self.south_arm.is_fixed_base}, jacobian_index: {self.south_jacobian_index}")

        self.north_jacobian_joint_ids = self._resolve_joint_indices(self.north_entity.joint_ids, self.north_arm)
        self.east_jacobian_joint_ids = self._resolve_joint_indices(self.east_entity.joint_ids, self.east_arm)
        self.west_jacobian_joint_ids = self._resolve_joint_indices(self.west_entity.joint_ids, self.west_arm)
        self.south_jacobian_joint_ids = self._resolve_joint_indices(self.south_entity.joint_ids, self.south_arm)

        if not self.north_arm.is_fixed_base:
            self.north_jacobian_joint_ids = [idx + 6 for idx in self.north_jacobian_joint_ids]
        if not self.east_arm.is_fixed_base:
            self.east_jacobian_joint_ids = [idx + 6 for idx in self.east_jacobian_joint_ids]
        if not self.west_arm.is_fixed_base:
            self.west_jacobian_joint_ids = [idx + 6 for idx in self.west_jacobian_joint_ids]
        if not self.south_arm.is_fixed_base:
            self.south_jacobian_joint_ids = [idx + 6 for idx in self.south_jacobian_joint_ids]

        # IK Verification: Print final joint indices
        print(f"\n[IK VERIFICATION] Jacobian Joint Indices:")
        print(f"  North: {self.north_jacobian_joint_ids}")
        print(f"  East:  {self.east_jacobian_joint_ids}")
        print(f"  West:  {self.west_jacobian_joint_ids}")
        print(f"  South: {self.south_jacobian_joint_ids}")
        print("="*80 + "\n")

        # State
        self.commanded_north_world = None
        self.commanded_east_world = None
        self.commanded_west_world = None
        self.commanded_south_world = None
        self.current_waypoint: Optional[WaypointCommand] = None
        self.hold_counter = 0
        self.required_hold = 0

    def _resolve_joint_indices(self, raw_ids, asset):
        """Resolve joint indices from slice or list."""
        if isinstance(raw_ids, slice):
            start = raw_ids.start or 0
            stop = raw_ids.stop or asset.num_joints
            step = raw_ids.step or 1
            return list(range(start, stop, step))
        return list(raw_ids)

    def reset(self):
        """Reset controller state."""
        self.commanded_north_world = None
        self.commanded_east_world = None
        self.commanded_west_world = None
        self.commanded_south_world = None
        self.current_waypoint = None
        self.hold_counter = 0
        self.required_hold = 0

    def set_waypoint(self, waypoint: WaypointCommand, hold_steps_override: Optional[int] = None):
        """Set new target waypoint."""
        self.current_waypoint = waypoint
        self.hold_counter = 0
        self.required_hold = hold_steps_override if hold_steps_override is not None else waypoint.hold_steps

        # Initialize commanded pose to current EE pose
        ee_pose_north_w = self.north_arm.data.body_pose_w[:, self.north_entity.body_ids[0]]
        ee_pose_east_w = self.east_arm.data.body_pose_w[:, self.east_entity.body_ids[0]]
        ee_pose_west_w = self.west_arm.data.body_pose_w[:, self.west_entity.body_ids[0]]
        ee_pose_south_w = self.south_arm.data.body_pose_w[:, self.south_entity.body_ids[0]]

        if self.commanded_north_world is None:
            self.commanded_north_world = ee_pose_north_w.clone()
            self.commanded_east_world = ee_pose_east_w.clone()
            self.commanded_west_world = ee_pose_west_w.clone()
            self.commanded_south_world = ee_pose_south_w.clone()
        else:
            self.commanded_north_world = ee_pose_north_w.clone()
            self.commanded_east_world = ee_pose_east_w.clone()
            self.commanded_west_world = ee_pose_west_w.clone()
            self.commanded_south_world = ee_pose_south_w.clone()

        self.commanded_north_world[:, 3:] = self._normalize_quat(self.commanded_north_world[:, 3:])
        self.commanded_east_world[:, 3:] = self._normalize_quat(self.commanded_east_world[:, 3:])
        self.commanded_west_world[:, 3:] = self._normalize_quat(self.commanded_west_world[:, 3:])
        self.commanded_south_world[:, 3:] = self._normalize_quat(self.commanded_south_world[:, 3:])

        # Reset IK controllers
        self.north_ik.reset()
        self.east_ik.reset()
        self.west_ik.reset()
        self.south_ik.reset()

        # Set initial IK command
        root_pose_north = self.north_arm.data.root_pose_w
        root_pose_east = self.east_arm.data.root_pose_w
        root_pose_west = self.west_arm.data.root_pose_w
        root_pose_south = self.south_arm.data.root_pose_w

        if self.controls_orientation:
            desired_north_base = self._convert_world_to_base(self.commanded_north_world.view(-1), root_pose_north).view(self.env.num_envs, -1)
            desired_east_base = self._convert_world_to_base(self.commanded_east_world.view(-1), root_pose_east).view(self.env.num_envs, -1)
            desired_west_base = self._convert_world_to_base(self.commanded_west_world.view(-1), root_pose_west).view(self.env.num_envs, -1)
            desired_south_base = self._convert_world_to_base(self.commanded_south_world.view(-1), root_pose_south).view(self.env.num_envs, -1)
            self.north_ik.set_command(desired_north_base)
            self.east_ik.set_command(desired_east_base)
            self.west_ik.set_command(desired_west_base)
            self.south_ik.set_command(desired_south_base)
        else:
            ee_pos_north_b, ee_quat_north_b = subtract_frame_transforms(
                root_pose_north[:, 0:3], root_pose_north[:, 3:7],
                ee_pose_north_w[:, 0:3], ee_pose_north_w[:, 3:7]
            )
            ee_pos_east_b, ee_quat_east_b = subtract_frame_transforms(
                root_pose_east[:, 0:3], root_pose_east[:, 3:7],
                ee_pose_east_w[:, 0:3], ee_pose_east_w[:, 3:7]
            )
            ee_pos_west_b, ee_quat_west_b = subtract_frame_transforms(
                root_pose_west[:, 0:3], root_pose_west[:, 3:7],
                ee_pose_west_w[:, 0:3], ee_pose_west_w[:, 3:7]
            )
            ee_pos_south_b, ee_quat_south_b = subtract_frame_transforms(
                root_pose_south[:, 0:3], root_pose_south[:, 3:7],
                ee_pose_south_w[:, 0:3], ee_pose_south_w[:, 3:7]
            )
            desired_north_pos_base = self._convert_world_to_base(self.commanded_north_world.view(-1), root_pose_north).view(self.env.num_envs, -1)[:, :3]
            desired_east_pos_base = self._convert_world_to_base(self.commanded_east_world.view(-1), root_pose_east).view(self.env.num_envs, -1)[:, :3]
            desired_west_pos_base = self._convert_world_to_base(self.commanded_west_world.view(-1), root_pose_west).view(self.env.num_envs, -1)[:, :3]
            desired_south_pos_base = self._convert_world_to_base(self.commanded_south_world.view(-1), root_pose_south).view(self.env.num_envs, -1)[:, :3]
            self.north_ik.set_command(desired_north_pos_base, ee_quat=ee_quat_north_b)
            self.east_ik.set_command(desired_east_pos_base, ee_quat=ee_quat_east_b)
            self.west_ik.set_command(desired_west_pos_base, ee_quat=ee_quat_west_b)
            self.south_ik.set_command(desired_south_pos_base, ee_quat=ee_quat_south_b)

    def step(self) -> tuple[torch.Tensor, bool]:
        """Execute one control step towards current waypoint. Returns: (action, converged)"""
        if self.current_waypoint is None:
            raise RuntimeError("No waypoint set. Call set_waypoint() first.")

        # Get current state
        root_pose_north = self.north_arm.data.root_pose_w
        root_pose_east = self.east_arm.data.root_pose_w
        root_pose_west = self.west_arm.data.root_pose_w
        root_pose_south = self.south_arm.data.root_pose_w

        ee_pose_north_w = self.north_arm.data.body_pose_w[:, self.north_entity.body_ids[0]]
        ee_pose_east_w = self.east_arm.data.body_pose_w[:, self.east_entity.body_ids[0]]
        ee_pose_west_w = self.west_arm.data.body_pose_w[:, self.west_entity.body_ids[0]]
        ee_pose_south_w = self.south_arm.data.body_pose_w[:, self.south_entity.body_ids[0]]

        # Get Jacobians
        jacobian_north_w = self.north_arm.root_physx_view.get_jacobians()[:, self.north_jacobian_index, :, self.north_jacobian_joint_ids]
        jacobian_east_w = self.east_arm.root_physx_view.get_jacobians()[:, self.east_jacobian_index, :, self.east_jacobian_joint_ids]
        jacobian_west_w = self.west_arm.root_physx_view.get_jacobians()[:, self.west_jacobian_index, :, self.west_jacobian_joint_ids]
        jacobian_south_w = self.south_arm.root_physx_view.get_jacobians()[:, self.south_jacobian_index, :, self.south_jacobian_joint_ids]

        # Transform to base frame
        ee_pos_north_b, ee_quat_north_b = subtract_frame_transforms(
            root_pose_north[:, 0:3], root_pose_north[:, 3:7],
            ee_pose_north_w[:, 0:3], ee_pose_north_w[:, 3:7]
        )
        ee_pos_east_b, ee_quat_east_b = subtract_frame_transforms(
            root_pose_east[:, 0:3], root_pose_east[:, 3:7],
            ee_pose_east_w[:, 0:3], ee_pose_east_w[:, 3:7]
        )
        ee_pos_west_b, ee_quat_west_b = subtract_frame_transforms(
            root_pose_west[:, 0:3], root_pose_west[:, 3:7],
            ee_pose_west_w[:, 0:3], ee_pose_west_w[:, 3:7]
        )
        ee_pos_south_b, ee_quat_south_b = subtract_frame_transforms(
            root_pose_south[:, 0:3], root_pose_south[:, 3:7],
            ee_pose_south_w[:, 0:3], ee_pose_south_w[:, 3:7]
        )

        # Transform Jacobians
        rot_north = matrix_from_quat(quat_inv(root_pose_north[:, 3:7]))
        rot_east = matrix_from_quat(quat_inv(root_pose_east[:, 3:7]))
        rot_west = matrix_from_quat(quat_inv(root_pose_west[:, 3:7]))
        rot_south = matrix_from_quat(quat_inv(root_pose_south[:, 3:7]))

        jacobian_north = jacobian_north_w.clone()
        jacobian_north[:, :3, :] = torch.bmm(rot_north, jacobian_north_w[:, :3, :])
        jacobian_north[:, 3:, :] = torch.bmm(rot_north, jacobian_north_w[:, 3:, :])

        jacobian_east = jacobian_east_w.clone()
        jacobian_east[:, :3, :] = torch.bmm(rot_east, jacobian_east_w[:, :3, :])
        jacobian_east[:, 3:, :] = torch.bmm(rot_east, jacobian_east_w[:, 3:, :])

        jacobian_west = jacobian_west_w.clone()
        jacobian_west[:, :3, :] = torch.bmm(rot_west, jacobian_west_w[:, :3, :])
        jacobian_west[:, 3:, :] = torch.bmm(rot_west, jacobian_west_w[:, 3:, :])

        jacobian_south = jacobian_south_w.clone()
        jacobian_south[:, :3, :] = torch.bmm(rot_south, jacobian_south_w[:, :3, :])
        jacobian_south[:, 3:, :] = torch.bmm(rot_south, jacobian_south_w[:, 3:, :])

        # Interpolate towards target
        if self.controls_orientation:
            self.commanded_north_world = torch.lerp(self.commanded_north_world, self.current_waypoint.north_world_pose, self.pose_interp_gain)
            self.commanded_east_world = torch.lerp(self.commanded_east_world, self.current_waypoint.east_world_pose, self.pose_interp_gain)
            self.commanded_west_world = torch.lerp(self.commanded_west_world, self.current_waypoint.west_world_pose, self.pose_interp_gain)
            self.commanded_south_world = torch.lerp(self.commanded_south_world, self.current_waypoint.south_world_pose, self.pose_interp_gain)

            self.commanded_north_world[:, 3:] = self._normalize_quat(self.commanded_north_world[:, 3:])
            self.commanded_east_world[:, 3:] = self._normalize_quat(self.commanded_east_world[:, 3:])
            self.commanded_west_world[:, 3:] = self._normalize_quat(self.commanded_west_world[:, 3:])
            self.commanded_south_world[:, 3:] = self._normalize_quat(self.commanded_south_world[:, 3:])

            desired_north_base = self._convert_world_to_base(self.commanded_north_world.view(-1), root_pose_north).view(self.env.num_envs, -1)
            desired_east_base = self._convert_world_to_base(self.commanded_east_world.view(-1), root_pose_east).view(self.env.num_envs, -1)
            desired_west_base = self._convert_world_to_base(self.commanded_west_world.view(-1), root_pose_west).view(self.env.num_envs, -1)
            desired_south_base = self._convert_world_to_base(self.commanded_south_world.view(-1), root_pose_south).view(self.env.num_envs, -1)

            self.north_ik.set_command(desired_north_base)
            self.east_ik.set_command(desired_east_base)
            self.west_ik.set_command(desired_west_base)
            self.south_ik.set_command(desired_south_base)
        else:
            commanded_north_pos = torch.lerp(self.commanded_north_world[:, :3], self.current_waypoint.north_world_pose[:, :3], self.pose_interp_gain)
            commanded_east_pos = torch.lerp(self.commanded_east_world[:, :3], self.current_waypoint.east_world_pose[:, :3], self.pose_interp_gain)
            commanded_west_pos = torch.lerp(self.commanded_west_world[:, :3], self.current_waypoint.west_world_pose[:, :3], self.pose_interp_gain)
            commanded_south_pos = torch.lerp(self.commanded_south_world[:, :3], self.current_waypoint.south_world_pose[:, :3], self.pose_interp_gain)

            self.commanded_north_world[:, :3] = commanded_north_pos
            self.commanded_east_world[:, :3] = commanded_east_pos
            self.commanded_west_world[:, :3] = commanded_west_pos
            self.commanded_south_world[:, :3] = commanded_south_pos

            desired_north_pos_base = self._convert_world_to_base(self.commanded_north_world.view(-1), root_pose_north).view(self.env.num_envs, -1)[:, :3]
            desired_east_pos_base = self._convert_world_to_base(self.commanded_east_world.view(-1), root_pose_east).view(self.env.num_envs, -1)[:, :3]
            desired_west_pos_base = self._convert_world_to_base(self.commanded_west_world.view(-1), root_pose_west).view(self.env.num_envs, -1)[:, :3]
            desired_south_pos_base = self._convert_world_to_base(self.commanded_south_world.view(-1), root_pose_south).view(self.env.num_envs, -1)[:, :3]

            self.north_ik.set_command(desired_north_pos_base, ee_quat=ee_quat_north_b)
            self.east_ik.set_command(desired_east_pos_base, ee_quat=ee_quat_east_b)
            self.west_ik.set_command(desired_west_pos_base, ee_quat=ee_quat_west_b)
            self.south_ik.set_command(desired_south_pos_base, ee_quat=ee_quat_south_b)

        # Compute IK
        joint_pos_north = self.north_arm.data.joint_pos[:, self.north_entity.joint_ids]
        joint_pos_east = self.east_arm.data.joint_pos[:, self.east_entity.joint_ids]
        joint_pos_west = self.west_arm.data.joint_pos[:, self.west_entity.joint_ids]
        joint_pos_south = self.south_arm.data.joint_pos[:, self.south_entity.joint_ids]

        if self.controls_orientation:
            joint_targets_north = self.north_ik.compute(ee_pos_north_b, ee_quat_north_b, jacobian_north, joint_pos_north)
            joint_targets_east = self.east_ik.compute(ee_pos_east_b, ee_quat_east_b, jacobian_east, joint_pos_east)
            joint_targets_west = self.west_ik.compute(ee_pos_west_b, ee_quat_west_b, jacobian_west, joint_pos_west)
            joint_targets_south = self.south_ik.compute(ee_pos_south_b, ee_quat_south_b, jacobian_south, joint_pos_south)
        else:
            joint_targets_north = self.north_ik.compute(ee_pos_north_b, ee_quat_north_b, jacobian_north[:, :3, :], joint_pos_north)
            joint_targets_east = self.east_ik.compute(ee_pos_east_b, ee_quat_east_b, jacobian_east[:, :3, :], joint_pos_east)
            joint_targets_west = self.west_ik.compute(ee_pos_west_b, ee_quat_west_b, jacobian_west[:, :3, :], joint_pos_west)
            joint_targets_south = self.south_ik.compute(ee_pos_south_b, ee_quat_south_b, jacobian_south[:, :3, :], joint_pos_south)

        # Apply interpolation gain
        if self.interp_gain < 1.0:
            joint_targets_north = joint_pos_north + self.interp_gain * (joint_targets_north - joint_pos_north)
            joint_targets_east = joint_pos_east + self.interp_gain * (joint_targets_east - joint_pos_east)
            joint_targets_west = joint_pos_west + self.interp_gain * (joint_targets_west - joint_pos_west)
            joint_targets_south = joint_pos_south + self.interp_gain * (joint_targets_south - joint_pos_south)

        # Apply wrist_flex from waypoint or global setting
        wrist_flex_idx = 3

        if self.current_waypoint.north_wrist_flex is not None:
            joint_targets_north[:, wrist_flex_idx] = self.current_waypoint.north_wrist_flex
        elif self.force_wrist_down:
            joint_targets_north[:, wrist_flex_idx] = self.wrist_flex_angle

        if self.current_waypoint.east_wrist_flex is not None:
            joint_targets_east[:, wrist_flex_idx] = self.current_waypoint.east_wrist_flex
        elif self.force_wrist_down:
            joint_targets_east[:, wrist_flex_idx] = self.wrist_flex_angle

        if self.current_waypoint.west_wrist_flex is not None:
            joint_targets_west[:, wrist_flex_idx] = self.current_waypoint.west_wrist_flex
        elif self.force_wrist_down:
            joint_targets_west[:, wrist_flex_idx] = self.wrist_flex_angle

        if self.current_waypoint.south_wrist_flex is not None:
            joint_targets_south[:, wrist_flex_idx] = self.current_waypoint.south_wrist_flex
        elif self.force_wrist_down:
            joint_targets_south[:, wrist_flex_idx] = self.wrist_flex_angle

        # Apply wrist_flex boundary constraints (clamp to range)
        if self.wrist_flex_min is not None or self.wrist_flex_max is not None:
            if self.wrist_flex_min is not None:
                joint_targets_north[:, wrist_flex_idx] = torch.clamp(joint_targets_north[:, wrist_flex_idx], min=self.wrist_flex_min)
                joint_targets_east[:, wrist_flex_idx] = torch.clamp(joint_targets_east[:, wrist_flex_idx], min=self.wrist_flex_min)
                joint_targets_west[:, wrist_flex_idx] = torch.clamp(joint_targets_west[:, wrist_flex_idx], min=self.wrist_flex_min)
                joint_targets_south[:, wrist_flex_idx] = torch.clamp(joint_targets_south[:, wrist_flex_idx], min=self.wrist_flex_min)
            if self.wrist_flex_max is not None:
                joint_targets_north[:, wrist_flex_idx] = torch.clamp(joint_targets_north[:, wrist_flex_idx], max=self.wrist_flex_max)
                joint_targets_east[:, wrist_flex_idx] = torch.clamp(joint_targets_east[:, wrist_flex_idx], max=self.wrist_flex_max)
                joint_targets_west[:, wrist_flex_idx] = torch.clamp(joint_targets_west[:, wrist_flex_idx], max=self.wrist_flex_max)
                joint_targets_south[:, wrist_flex_idx] = torch.clamp(joint_targets_south[:, wrist_flex_idx], max=self.wrist_flex_max)

        # IK Validation: Check joint position delta (print every 30 steps for debugging)
        if hasattr(self, '_ik_debug_counter'):
            self._ik_debug_counter += 1
        else:
            self._ik_debug_counter = 0

        if self._ik_debug_counter % 30 == 0:
            joint_delta_north = torch.abs(joint_targets_north - joint_pos_north).max()
            joint_delta_east = torch.abs(joint_targets_east - joint_pos_east).max()
            joint_delta_west = torch.abs(joint_targets_west - joint_pos_west).max()
            joint_delta_south = torch.abs(joint_targets_south - joint_pos_south).max()
            max_delta = max(joint_delta_north.item(), joint_delta_east.item(), joint_delta_west.item(), joint_delta_south.item())

            if max_delta > 0.5:  # Large joint delta (>0.5 rad = 28.6 deg) indicates potential IK issue
                print(f"[IK WARNING] Large joint delta detected: {max_delta:.3f} rad")
                print(f"  North delta: {joint_delta_north.item():.3f} rad")
                print(f"  East delta:  {joint_delta_east.item():.3f} rad")
                print(f"  West delta:  {joint_delta_west.item():.3f} rad")
                print(f"  South delta: {joint_delta_south.item():.3f} rad")

        # Compose action
        action = self._compose_quad_arm_action(
            joint_targets_north, self.current_waypoint.north_gripper,
            joint_targets_east, self.current_waypoint.east_gripper,
            joint_targets_west, self.current_waypoint.west_gripper,
            joint_targets_south, self.current_waypoint.south_gripper
        )

        # Check convergence
        pos_err_north = torch.linalg.norm(self.current_waypoint.north_world_pose[:, :3] - ee_pose_north_w[:, :3], dim=-1)
        pos_err_east = torch.linalg.norm(self.current_waypoint.east_world_pose[:, :3] - ee_pose_east_w[:, :3], dim=-1)
        pos_err_west = torch.linalg.norm(self.current_waypoint.west_world_pose[:, :3] - ee_pose_west_w[:, :3], dim=-1)
        pos_err_south = torch.linalg.norm(self.current_waypoint.south_world_pose[:, :3] - ee_pose_south_w[:, :3], dim=-1)

        within_pos = (torch.all(pos_err_north <= self.position_tol) and
                      torch.all(pos_err_east <= self.position_tol) and
                      torch.all(pos_err_west <= self.position_tol) and
                      torch.all(pos_err_south <= self.position_tol))

        if self.controls_orientation:
            quat_err_north = 1.0 - torch.abs(torch.sum(self.current_waypoint.north_world_pose[:, 3:] * ee_pose_north_w[:, 3:], dim=-1))
            quat_err_east = 1.0 - torch.abs(torch.sum(self.current_waypoint.east_world_pose[:, 3:] * ee_pose_east_w[:, 3:], dim=-1))
            quat_err_west = 1.0 - torch.abs(torch.sum(self.current_waypoint.west_world_pose[:, 3:] * ee_pose_west_w[:, 3:], dim=-1))
            quat_err_south = 1.0 - torch.abs(torch.sum(self.current_waypoint.south_world_pose[:, 3:] * ee_pose_south_w[:, 3:], dim=-1))

            within_rot = (torch.all(quat_err_north <= self.orientation_tol) and
                         torch.all(quat_err_east <= self.orientation_tol) and
                         torch.all(quat_err_west <= self.orientation_tol) and
                         torch.all(quat_err_south <= self.orientation_tol))
        else:
            within_rot = True

        if within_pos and within_rot:
            self.hold_counter += 1
        else:
            # Print error if close to converging (for debugging)
            if self.hold_counter > 0:
                max_err = max(pos_err_north.item(), pos_err_east.item(), pos_err_west.item(), pos_err_south.item())
                print(f"[Convergence] Hold counter reset. Max position error: {max_err:.4f}m (tol: {self.position_tol})")
                print(f"  Individual errors: North={pos_err_north.item():.4f}m, East={pos_err_east.item():.4f}m, West={pos_err_west.item():.4f}m, South={pos_err_south.item():.4f}m")
            self.hold_counter = 0

        converged = self.hold_counter >= self.required_hold

        # Print progress when getting close
        if self.hold_counter > 0 and self.hold_counter % 10 == 0:
            print(f"[Convergence] Hold counter: {self.hold_counter}/{self.required_hold}")

        # Debug: Print current robot positions when converged
        if converged:
            # Extra IK verification on first waypoint convergence
            if not hasattr(self, '_first_waypoint_verified'):
                self._first_waypoint_verified = True
                print("\n" + "="*80)
                print("[IK VERIFICATION] First Waypoint Convergence - Detailed Check")
                print("="*80)
                print(f"Expected body: 'gripper' | Checking if EE position matches gripper tip...")
                print(f"\nRobot Base Positions:")
                print(f"  North base: ({root_pose_north[0, 0]:.4f}, {root_pose_north[0, 1]:.4f}, {root_pose_north[0, 2]:.4f})")
                print(f"  East  base: ({root_pose_east[0, 0]:.4f}, {root_pose_east[0, 1]:.4f}, {root_pose_east[0, 2]:.4f})")
                print(f"  West  base: ({root_pose_west[0, 0]:.4f}, {root_pose_west[0, 1]:.4f}, {root_pose_west[0, 2]:.4f})")
                print(f"  South base: ({root_pose_south[0, 0]:.4f}, {root_pose_south[0, 1]:.4f}, {root_pose_south[0, 2]:.4f})")
                print(f"\nEE Position in Base Frame:")
                print(f"  North: ({ee_pos_north_b[0, 0]:.4f}, {ee_pos_north_b[0, 1]:.4f}, {ee_pos_north_b[0, 2]:.4f})")
                print(f"  East:  ({ee_pos_east_b[0, 0]:.4f}, {ee_pos_east_b[0, 1]:.4f}, {ee_pos_east_b[0, 2]:.4f})")
                print(f"  West:  ({ee_pos_west_b[0, 0]:.4f}, {ee_pos_west_b[0, 1]:.4f}, {ee_pos_west_b[0, 2]:.4f})")
                print(f"  South: ({ee_pos_south_b[0, 0]:.4f}, {ee_pos_south_b[0, 1]:.4f}, {ee_pos_south_b[0, 2]:.4f})")
                print("="*80 + "\n")

            print(f"\n[DEBUG] Waypoint CONVERGED - Current Robot Positions:")
            print(f"  North EE: x={ee_pose_north_w[0, 0]:.4f}, y={ee_pose_north_w[0, 1]:.4f}, z={ee_pose_north_w[0, 2]:.4f}")
            print(f"  East  EE: x={ee_pose_east_w[0, 0]:.4f}, y={ee_pose_east_w[0, 1]:.4f}, z={ee_pose_east_w[0, 2]:.4f}")
            print(f"  West  EE: x={ee_pose_west_w[0, 0]:.4f}, y={ee_pose_west_w[0, 1]:.4f}, z={ee_pose_west_w[0, 2]:.4f}")
            print(f"  South EE: x={ee_pose_south_w[0, 0]:.4f}, y={ee_pose_south_w[0, 1]:.4f}, z={ee_pose_south_w[0, 2]:.4f}")
            print(f"[DEBUG] Target Waypoint Positions:")
            print(f"  North Target: x={self.current_waypoint.north_world_pose[0, 0]:.4f}, y={self.current_waypoint.north_world_pose[0, 1]:.4f}, z={self.current_waypoint.north_world_pose[0, 2]:.4f}")
            print(f"  East  Target: x={self.current_waypoint.east_world_pose[0, 0]:.4f}, y={self.current_waypoint.east_world_pose[0, 1]:.4f}, z={self.current_waypoint.east_world_pose[0, 2]:.4f}")
            print(f"  West  Target: x={self.current_waypoint.west_world_pose[0, 0]:.4f}, y={self.current_waypoint.west_world_pose[0, 1]:.4f}, z={self.current_waypoint.west_world_pose[0, 2]:.4f}")
            print(f"  South Target: x={self.current_waypoint.south_world_pose[0, 0]:.4f}, y={self.current_waypoint.south_world_pose[0, 1]:.4f}, z={self.current_waypoint.south_world_pose[0, 2]:.4f}")
            print(f"[DEBUG] Position Errors:")
            print(f"  North: {pos_err_north.item():.4f}m")
            print(f"  East:  {pos_err_east.item():.4f}m")
            print(f"  West:  {pos_err_west.item():.4f}m")
            print(f"  South: {pos_err_south.item():.4f}m")

            # IK Verification: Print joint positions
            print(f"[DEBUG] Current Joint Positions (rad):")
            print(f"  North joints: {joint_pos_north[0, :5].cpu().numpy()}")
            print(f"  East  joints: {joint_pos_east[0, :5].cpu().numpy()}")
            print(f"  West  joints: {joint_pos_west[0, :5].cpu().numpy()}")
            print(f"  South joints: {joint_pos_south[0, :5].cpu().numpy()}")

            # Check if any joints are at limits (potential issue)
            joint_limits_hit = []
            if (torch.abs(joint_pos_north) > 1.5).any():
                joint_limits_hit.append("North")
            if (torch.abs(joint_pos_east) > 1.5).any():
                joint_limits_hit.append("East")
            if (torch.abs(joint_pos_west) > 1.5).any():
                joint_limits_hit.append("West")
            if (torch.abs(joint_pos_south) > 1.5).any():
                joint_limits_hit.append("South")

            if joint_limits_hit:
                print(f"[IK WARNING] Joints near limits detected: {', '.join(joint_limits_hit)}")
            print()

        return action, converged

    def _normalize_quat(self, quat):
        """Normalize quaternion."""
        return quat / torch.linalg.norm(quat, dim=-1, keepdim=True)

    def _convert_world_to_base(self, world_pose, root_pose):
        """Convert world-frame pose to base-frame."""
        if world_pose.dim() == 1:
            pos_w = world_pose[:3]
            quat_w = world_pose[3:7]
        else:
            pos_w = world_pose[0, :3]
            quat_w = world_pose[0, 3:7]

        pos_b, quat_b = subtract_frame_transforms(
            root_pose[:, 0:3], root_pose[:, 3:7], pos_w.unsqueeze(0), quat_w.unsqueeze(0)
        )
        return torch.cat([pos_b, quat_b], dim=-1)

    def _compose_quad_arm_action(self, north_joints, north_gripper, east_joints, east_gripper,
                                  west_joints, west_gripper, south_joints, south_gripper):
        """Compose 24-DOF quad-arm action tensor."""
        action = torch.zeros(self.env.num_envs, 24, device=self.env.device)
        action[:, :5] = north_joints[:, :5]
        action[:, 5] = north_gripper
        action[:, 6:11] = east_joints[:, :5]
        action[:, 11] = east_gripper
        action[:, 12:17] = west_joints[:, :5]
        action[:, 17] = west_gripper
        action[:, 18:23] = south_joints[:, :5]
        action[:, 23] = south_gripper
        return action


def load_waypoints_from_json(filepath: str, device: str) -> list[WaypointCommand]:
    """Load waypoints from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Waypoint file must contain a JSON array")

    waypoints = []
    for idx, wp in enumerate(data):
        if "north" not in wp or "east" not in wp or "west" not in wp or "south" not in wp:
            raise ValueError(f"Waypoint {idx} missing arm data")

        north_pose = _parse_waypoint_pose(wp["north"], device)
        east_pose = _parse_waypoint_pose(wp["east"], device)
        west_pose = _parse_waypoint_pose(wp["west"], device)
        south_pose = _parse_waypoint_pose(wp["south"], device)

        waypoints.append(WaypointCommand(
            north_world_pose=north_pose.unsqueeze(0),
            east_world_pose=east_pose.unsqueeze(0),
            west_world_pose=west_pose.unsqueeze(0),
            south_world_pose=south_pose.unsqueeze(0),
            north_gripper=wp["north"].get("gripper", 0.0),
            east_gripper=wp["east"].get("gripper", 0.0),
            west_gripper=wp["west"].get("gripper", 0.0),
            south_gripper=wp["south"].get("gripper", 0.0),
            hold_steps=wp.get("hold_steps", 30),
            north_wrist_flex=wp["north"].get("wrist_flex", None),
            east_wrist_flex=wp["east"].get("wrist_flex", None),
            west_wrist_flex=wp["west"].get("wrist_flex", None),
            south_wrist_flex=wp["south"].get("wrist_flex", None),
        ))

    return waypoints


def _parse_waypoint_pose(pose: dict, device: str, default_orientation: list = None) -> torch.Tensor:
    """Parse position and orientation from waypoint dict."""
    if "position" not in pose:
        raise ValueError("Waypoint must have 'position' field")

    position = torch.as_tensor(pose["position"], dtype=torch.float32, device=device)
    if position.shape != (3,):
        raise ValueError(f"Position must be 3D, got {position.shape}")

    if "orientation" in pose:
        orientation = torch.as_tensor(pose["orientation"], dtype=torch.float32, device=device)
    elif default_orientation is not None:
        orientation = torch.as_tensor(default_orientation, dtype=torch.float32, device=device)
    else:
        orientation = torch.tensor(IDENTITY_QUAT, dtype=torch.float32, device=device)

    if orientation.shape != (4,):
        raise ValueError(f"Orientation must be 4D quaternion, got {orientation.shape}")

    orientation = orientation / torch.linalg.norm(orientation)

    return torch.cat([position, orientation], dim=0)

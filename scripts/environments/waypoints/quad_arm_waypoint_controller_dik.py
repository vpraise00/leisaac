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
    nord_world_pose: torch.Tensor  # (1, 7) - position + quaternion
    ost_world_pose: torch.Tensor  # (1, 7)
    west_world_pose: torch.Tensor  # (1, 7)
    sud_world_pose: torch.Tensor  # (1, 7)
    nord_gripper: float
    ost_gripper: float
    west_gripper: float
    sud_gripper: float
    hold_steps: int
    nord_wrist_flex: Optional[float] = None
    ost_wrist_flex: Optional[float] = None
    west_wrist_flex: Optional[float] = None
    sud_wrist_flex: Optional[float] = None


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
        force_wrist_down: bool = True,
        wrist_flex_angle: float = 1.57,
    ):
        """Initialize quad-arm waypoint controller."""
        self.env = env
        self.position_tol = position_tol
        self.orientation_tol = orientation_tol
        self.pose_interp_gain = pose_interp_gain
        self.interp_gain = interp_gain
        self.force_wrist_down = force_wrist_down
        self.wrist_flex_angle = wrist_flex_angle

        # Setup arms
        self.nord_arm = env.scene["nord_arm"]
        self.ost_arm = env.scene["ost_arm"]
        self.west_arm = env.scene["west_arm"]
        self.sud_arm = env.scene["sud_arm"]

        self.nord_entity = SceneEntityCfg("nord_arm", joint_names=ARM_JOINT_NAMES, body_names=["gripper"])
        self.ost_entity = SceneEntityCfg("ost_arm", joint_names=ARM_JOINT_NAMES, body_names=["gripper"])
        self.west_entity = SceneEntityCfg("west_arm", joint_names=ARM_JOINT_NAMES, body_names=["gripper"])
        self.sud_entity = SceneEntityCfg("sud_arm", joint_names=ARM_JOINT_NAMES, body_names=["gripper"])

        self.nord_entity.resolve(env.scene)
        self.ost_entity.resolve(env.scene)
        self.west_entity.resolve(env.scene)
        self.sud_entity.resolve(env.scene)

        # Setup IK controllers
        ik_cfg = DifferentialIKControllerCfg(command_type=command_type, use_relative_mode=False, ik_method=ik_method)
        self.nord_ik = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)
        self.ost_ik = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)
        self.west_ik = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)
        self.sud_ik = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)
        self.controls_orientation = ik_cfg.command_type == "pose"

        # Setup Jacobian indices
        self.nord_jacobian_index = self.nord_entity.body_ids[0] - 1 if self.nord_arm.is_fixed_base else self.nord_entity.body_ids[0]
        self.ost_jacobian_index = self.ost_entity.body_ids[0] - 1 if self.ost_arm.is_fixed_base else self.ost_entity.body_ids[0]
        self.west_jacobian_index = self.west_entity.body_ids[0] - 1 if self.west_arm.is_fixed_base else self.west_entity.body_ids[0]
        self.sud_jacobian_index = self.sud_entity.body_ids[0] - 1 if self.sud_arm.is_fixed_base else self.sud_entity.body_ids[0]

        self.nord_jacobian_joint_ids = self._resolve_joint_indices(self.nord_entity.joint_ids, self.nord_arm)
        self.ost_jacobian_joint_ids = self._resolve_joint_indices(self.ost_entity.joint_ids, self.ost_arm)
        self.west_jacobian_joint_ids = self._resolve_joint_indices(self.west_entity.joint_ids, self.west_arm)
        self.sud_jacobian_joint_ids = self._resolve_joint_indices(self.sud_entity.joint_ids, self.sud_arm)

        if not self.nord_arm.is_fixed_base:
            self.nord_jacobian_joint_ids = [idx + 6 for idx in self.nord_jacobian_joint_ids]
        if not self.ost_arm.is_fixed_base:
            self.ost_jacobian_joint_ids = [idx + 6 for idx in self.ost_jacobian_joint_ids]
        if not self.west_arm.is_fixed_base:
            self.west_jacobian_joint_ids = [idx + 6 for idx in self.west_jacobian_joint_ids]
        if not self.sud_arm.is_fixed_base:
            self.sud_jacobian_joint_ids = [idx + 6 for idx in self.sud_jacobian_joint_ids]

        # State
        self.commanded_nord_world = None
        self.commanded_ost_world = None
        self.commanded_west_world = None
        self.commanded_sud_world = None
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
        self.commanded_nord_world = None
        self.commanded_ost_world = None
        self.commanded_west_world = None
        self.commanded_sud_world = None
        self.current_waypoint = None
        self.hold_counter = 0
        self.required_hold = 0

    def set_waypoint(self, waypoint: WaypointCommand, hold_steps_override: Optional[int] = None):
        """Set new target waypoint."""
        self.current_waypoint = waypoint
        self.hold_counter = 0
        self.required_hold = hold_steps_override if hold_steps_override is not None else waypoint.hold_steps

        # Initialize commanded pose to current EE pose
        ee_pose_nord_w = self.nord_arm.data.body_pose_w[:, self.nord_entity.body_ids[0]]
        ee_pose_ost_w = self.ost_arm.data.body_pose_w[:, self.ost_entity.body_ids[0]]
        ee_pose_west_w = self.west_arm.data.body_pose_w[:, self.west_entity.body_ids[0]]
        ee_pose_sud_w = self.sud_arm.data.body_pose_w[:, self.sud_entity.body_ids[0]]

        if self.commanded_nord_world is None:
            self.commanded_nord_world = ee_pose_nord_w.clone()
            self.commanded_ost_world = ee_pose_ost_w.clone()
            self.commanded_west_world = ee_pose_west_w.clone()
            self.commanded_sud_world = ee_pose_sud_w.clone()
        else:
            self.commanded_nord_world = ee_pose_nord_w.clone()
            self.commanded_ost_world = ee_pose_ost_w.clone()
            self.commanded_west_world = ee_pose_west_w.clone()
            self.commanded_sud_world = ee_pose_sud_w.clone()

        self.commanded_nord_world[:, 3:] = self._normalize_quat(self.commanded_nord_world[:, 3:])
        self.commanded_ost_world[:, 3:] = self._normalize_quat(self.commanded_ost_world[:, 3:])
        self.commanded_west_world[:, 3:] = self._normalize_quat(self.commanded_west_world[:, 3:])
        self.commanded_sud_world[:, 3:] = self._normalize_quat(self.commanded_sud_world[:, 3:])

        # Reset IK controllers
        self.nord_ik.reset()
        self.ost_ik.reset()
        self.west_ik.reset()
        self.sud_ik.reset()

        # Set initial IK command
        root_pose_nord = self.nord_arm.data.root_pose_w
        root_pose_ost = self.ost_arm.data.root_pose_w
        root_pose_west = self.west_arm.data.root_pose_w
        root_pose_sud = self.sud_arm.data.root_pose_w

        if self.controls_orientation:
            desired_nord_base = self._convert_world_to_base(self.commanded_nord_world.view(-1), root_pose_nord).view(self.env.num_envs, -1)
            desired_ost_base = self._convert_world_to_base(self.commanded_ost_world.view(-1), root_pose_ost).view(self.env.num_envs, -1)
            desired_west_base = self._convert_world_to_base(self.commanded_west_world.view(-1), root_pose_west).view(self.env.num_envs, -1)
            desired_sud_base = self._convert_world_to_base(self.commanded_sud_world.view(-1), root_pose_sud).view(self.env.num_envs, -1)
            self.nord_ik.set_command(desired_nord_base)
            self.ost_ik.set_command(desired_ost_base)
            self.west_ik.set_command(desired_west_base)
            self.sud_ik.set_command(desired_sud_base)
        else:
            ee_pos_nord_b, ee_quat_nord_b = subtract_frame_transforms(
                root_pose_nord[:, 0:3], root_pose_nord[:, 3:7],
                ee_pose_nord_w[:, 0:3], ee_pose_nord_w[:, 3:7]
            )
            ee_pos_ost_b, ee_quat_ost_b = subtract_frame_transforms(
                root_pose_ost[:, 0:3], root_pose_ost[:, 3:7],
                ee_pose_ost_w[:, 0:3], ee_pose_ost_w[:, 3:7]
            )
            ee_pos_west_b, ee_quat_west_b = subtract_frame_transforms(
                root_pose_west[:, 0:3], root_pose_west[:, 3:7],
                ee_pose_west_w[:, 0:3], ee_pose_west_w[:, 3:7]
            )
            ee_pos_sud_b, ee_quat_sud_b = subtract_frame_transforms(
                root_pose_sud[:, 0:3], root_pose_sud[:, 3:7],
                ee_pose_sud_w[:, 0:3], ee_pose_sud_w[:, 3:7]
            )
            desired_nord_pos_base = self._convert_world_to_base(self.commanded_nord_world.view(-1), root_pose_nord).view(self.env.num_envs, -1)[:, :3]
            desired_ost_pos_base = self._convert_world_to_base(self.commanded_ost_world.view(-1), root_pose_ost).view(self.env.num_envs, -1)[:, :3]
            desired_west_pos_base = self._convert_world_to_base(self.commanded_west_world.view(-1), root_pose_west).view(self.env.num_envs, -1)[:, :3]
            desired_sud_pos_base = self._convert_world_to_base(self.commanded_sud_world.view(-1), root_pose_sud).view(self.env.num_envs, -1)[:, :3]
            self.nord_ik.set_command(desired_nord_pos_base, ee_quat=ee_quat_nord_b)
            self.ost_ik.set_command(desired_ost_pos_base, ee_quat=ee_quat_ost_b)
            self.west_ik.set_command(desired_west_pos_base, ee_quat=ee_quat_west_b)
            self.sud_ik.set_command(desired_sud_pos_base, ee_quat=ee_quat_sud_b)

    def step(self) -> tuple[torch.Tensor, bool]:
        """Execute one control step towards current waypoint. Returns: (action, converged)"""
        if self.current_waypoint is None:
            raise RuntimeError("No waypoint set. Call set_waypoint() first.")

        # Get current state
        root_pose_nord = self.nord_arm.data.root_pose_w
        root_pose_ost = self.ost_arm.data.root_pose_w
        root_pose_west = self.west_arm.data.root_pose_w
        root_pose_sud = self.sud_arm.data.root_pose_w

        ee_pose_nord_w = self.nord_arm.data.body_pose_w[:, self.nord_entity.body_ids[0]]
        ee_pose_ost_w = self.ost_arm.data.body_pose_w[:, self.ost_entity.body_ids[0]]
        ee_pose_west_w = self.west_arm.data.body_pose_w[:, self.west_entity.body_ids[0]]
        ee_pose_sud_w = self.sud_arm.data.body_pose_w[:, self.sud_entity.body_ids[0]]

        # Get Jacobians
        jacobian_nord_w = self.nord_arm.root_physx_view.get_jacobians()[:, self.nord_jacobian_index, :, self.nord_jacobian_joint_ids]
        jacobian_ost_w = self.ost_arm.root_physx_view.get_jacobians()[:, self.ost_jacobian_index, :, self.ost_jacobian_joint_ids]
        jacobian_west_w = self.west_arm.root_physx_view.get_jacobians()[:, self.west_jacobian_index, :, self.west_jacobian_joint_ids]
        jacobian_sud_w = self.sud_arm.root_physx_view.get_jacobians()[:, self.sud_jacobian_index, :, self.sud_jacobian_joint_ids]

        # Transform to base frame
        ee_pos_nord_b, ee_quat_nord_b = subtract_frame_transforms(
            root_pose_nord[:, 0:3], root_pose_nord[:, 3:7],
            ee_pose_nord_w[:, 0:3], ee_pose_nord_w[:, 3:7]
        )
        ee_pos_ost_b, ee_quat_ost_b = subtract_frame_transforms(
            root_pose_ost[:, 0:3], root_pose_ost[:, 3:7],
            ee_pose_ost_w[:, 0:3], ee_pose_ost_w[:, 3:7]
        )
        ee_pos_west_b, ee_quat_west_b = subtract_frame_transforms(
            root_pose_west[:, 0:3], root_pose_west[:, 3:7],
            ee_pose_west_w[:, 0:3], ee_pose_west_w[:, 3:7]
        )
        ee_pos_sud_b, ee_quat_sud_b = subtract_frame_transforms(
            root_pose_sud[:, 0:3], root_pose_sud[:, 3:7],
            ee_pose_sud_w[:, 0:3], ee_pose_sud_w[:, 3:7]
        )

        # Transform Jacobians
        rot_nord = matrix_from_quat(quat_inv(root_pose_nord[:, 3:7]))
        rot_ost = matrix_from_quat(quat_inv(root_pose_ost[:, 3:7]))
        rot_west = matrix_from_quat(quat_inv(root_pose_west[:, 3:7]))
        rot_sud = matrix_from_quat(quat_inv(root_pose_sud[:, 3:7]))

        jacobian_nord = jacobian_nord_w.clone()
        jacobian_nord[:, :3, :] = torch.bmm(rot_nord, jacobian_nord_w[:, :3, :])
        jacobian_nord[:, 3:, :] = torch.bmm(rot_nord, jacobian_nord_w[:, 3:, :])

        jacobian_ost = jacobian_ost_w.clone()
        jacobian_ost[:, :3, :] = torch.bmm(rot_ost, jacobian_ost_w[:, :3, :])
        jacobian_ost[:, 3:, :] = torch.bmm(rot_ost, jacobian_ost_w[:, 3:, :])

        jacobian_west = jacobian_west_w.clone()
        jacobian_west[:, :3, :] = torch.bmm(rot_west, jacobian_west_w[:, :3, :])
        jacobian_west[:, 3:, :] = torch.bmm(rot_west, jacobian_west_w[:, 3:, :])

        jacobian_sud = jacobian_sud_w.clone()
        jacobian_sud[:, :3, :] = torch.bmm(rot_sud, jacobian_sud_w[:, :3, :])
        jacobian_sud[:, 3:, :] = torch.bmm(rot_sud, jacobian_sud_w[:, 3:, :])

        # Interpolate towards target
        if self.controls_orientation:
            self.commanded_nord_world = torch.lerp(self.commanded_nord_world, self.current_waypoint.nord_world_pose, self.pose_interp_gain)
            self.commanded_ost_world = torch.lerp(self.commanded_ost_world, self.current_waypoint.ost_world_pose, self.pose_interp_gain)
            self.commanded_west_world = torch.lerp(self.commanded_west_world, self.current_waypoint.west_world_pose, self.pose_interp_gain)
            self.commanded_sud_world = torch.lerp(self.commanded_sud_world, self.current_waypoint.sud_world_pose, self.pose_interp_gain)

            self.commanded_nord_world[:, 3:] = self._normalize_quat(self.commanded_nord_world[:, 3:])
            self.commanded_ost_world[:, 3:] = self._normalize_quat(self.commanded_ost_world[:, 3:])
            self.commanded_west_world[:, 3:] = self._normalize_quat(self.commanded_west_world[:, 3:])
            self.commanded_sud_world[:, 3:] = self._normalize_quat(self.commanded_sud_world[:, 3:])

            desired_nord_base = self._convert_world_to_base(self.commanded_nord_world.view(-1), root_pose_nord).view(self.env.num_envs, -1)
            desired_ost_base = self._convert_world_to_base(self.commanded_ost_world.view(-1), root_pose_ost).view(self.env.num_envs, -1)
            desired_west_base = self._convert_world_to_base(self.commanded_west_world.view(-1), root_pose_west).view(self.env.num_envs, -1)
            desired_sud_base = self._convert_world_to_base(self.commanded_sud_world.view(-1), root_pose_sud).view(self.env.num_envs, -1)

            self.nord_ik.set_command(desired_nord_base)
            self.ost_ik.set_command(desired_ost_base)
            self.west_ik.set_command(desired_west_base)
            self.sud_ik.set_command(desired_sud_base)
        else:
            commanded_nord_pos = torch.lerp(self.commanded_nord_world[:, :3], self.current_waypoint.nord_world_pose[:, :3], self.pose_interp_gain)
            commanded_ost_pos = torch.lerp(self.commanded_ost_world[:, :3], self.current_waypoint.ost_world_pose[:, :3], self.pose_interp_gain)
            commanded_west_pos = torch.lerp(self.commanded_west_world[:, :3], self.current_waypoint.west_world_pose[:, :3], self.pose_interp_gain)
            commanded_sud_pos = torch.lerp(self.commanded_sud_world[:, :3], self.current_waypoint.sud_world_pose[:, :3], self.pose_interp_gain)

            self.commanded_nord_world[:, :3] = commanded_nord_pos
            self.commanded_ost_world[:, :3] = commanded_ost_pos
            self.commanded_west_world[:, :3] = commanded_west_pos
            self.commanded_sud_world[:, :3] = commanded_sud_pos

            desired_nord_pos_base = self._convert_world_to_base(self.commanded_nord_world.view(-1), root_pose_nord).view(self.env.num_envs, -1)[:, :3]
            desired_ost_pos_base = self._convert_world_to_base(self.commanded_ost_world.view(-1), root_pose_ost).view(self.env.num_envs, -1)[:, :3]
            desired_west_pos_base = self._convert_world_to_base(self.commanded_west_world.view(-1), root_pose_west).view(self.env.num_envs, -1)[:, :3]
            desired_sud_pos_base = self._convert_world_to_base(self.commanded_sud_world.view(-1), root_pose_sud).view(self.env.num_envs, -1)[:, :3]

            self.nord_ik.set_command(desired_nord_pos_base, ee_quat=ee_quat_nord_b)
            self.ost_ik.set_command(desired_ost_pos_base, ee_quat=ee_quat_ost_b)
            self.west_ik.set_command(desired_west_pos_base, ee_quat=ee_quat_west_b)
            self.sud_ik.set_command(desired_sud_pos_base, ee_quat=ee_quat_sud_b)

        # Compute IK
        joint_pos_nord = self.nord_arm.data.joint_pos[:, self.nord_entity.joint_ids]
        joint_pos_ost = self.ost_arm.data.joint_pos[:, self.ost_entity.joint_ids]
        joint_pos_west = self.west_arm.data.joint_pos[:, self.west_entity.joint_ids]
        joint_pos_sud = self.sud_arm.data.joint_pos[:, self.sud_entity.joint_ids]

        if self.controls_orientation:
            joint_targets_nord = self.nord_ik.compute(ee_pos_nord_b, ee_quat_nord_b, jacobian_nord, joint_pos_nord)
            joint_targets_ost = self.ost_ik.compute(ee_pos_ost_b, ee_quat_ost_b, jacobian_ost, joint_pos_ost)
            joint_targets_west = self.west_ik.compute(ee_pos_west_b, ee_quat_west_b, jacobian_west, joint_pos_west)
            joint_targets_sud = self.sud_ik.compute(ee_pos_sud_b, ee_quat_sud_b, jacobian_sud, joint_pos_sud)
        else:
            joint_targets_nord = self.nord_ik.compute(ee_pos_nord_b, ee_quat_nord_b, jacobian_nord[:, :3, :], joint_pos_nord)
            joint_targets_ost = self.ost_ik.compute(ee_pos_ost_b, ee_quat_ost_b, jacobian_ost[:, :3, :], joint_pos_ost)
            joint_targets_west = self.west_ik.compute(ee_pos_west_b, ee_quat_west_b, jacobian_west[:, :3, :], joint_pos_west)
            joint_targets_sud = self.sud_ik.compute(ee_pos_sud_b, ee_quat_sud_b, jacobian_sud[:, :3, :], joint_pos_sud)

        # Apply interpolation gain
        if self.interp_gain < 1.0:
            joint_targets_nord = joint_pos_nord + self.interp_gain * (joint_targets_nord - joint_pos_nord)
            joint_targets_ost = joint_pos_ost + self.interp_gain * (joint_targets_ost - joint_pos_ost)
            joint_targets_west = joint_pos_west + self.interp_gain * (joint_targets_west - joint_pos_west)
            joint_targets_sud = joint_pos_sud + self.interp_gain * (joint_targets_sud - joint_pos_sud)

        # Apply wrist_flex from waypoint or global setting
        wrist_flex_idx = 3

        if self.current_waypoint.nord_wrist_flex is not None:
            joint_targets_nord[:, wrist_flex_idx] = self.current_waypoint.nord_wrist_flex
        elif self.force_wrist_down:
            joint_targets_nord[:, wrist_flex_idx] = self.wrist_flex_angle

        if self.current_waypoint.ost_wrist_flex is not None:
            joint_targets_ost[:, wrist_flex_idx] = self.current_waypoint.ost_wrist_flex
        elif self.force_wrist_down:
            joint_targets_ost[:, wrist_flex_idx] = self.wrist_flex_angle

        if self.current_waypoint.west_wrist_flex is not None:
            joint_targets_west[:, wrist_flex_idx] = self.current_waypoint.west_wrist_flex
        elif self.force_wrist_down:
            joint_targets_west[:, wrist_flex_idx] = self.wrist_flex_angle

        if self.current_waypoint.sud_wrist_flex is not None:
            joint_targets_sud[:, wrist_flex_idx] = self.current_waypoint.sud_wrist_flex
        elif self.force_wrist_down:
            joint_targets_sud[:, wrist_flex_idx] = self.wrist_flex_angle

        # Compose action
        action = self._compose_quad_arm_action(
            joint_targets_nord, self.current_waypoint.nord_gripper,
            joint_targets_ost, self.current_waypoint.ost_gripper,
            joint_targets_west, self.current_waypoint.west_gripper,
            joint_targets_sud, self.current_waypoint.sud_gripper
        )

        # Check convergence
        pos_err_nord = torch.linalg.norm(self.current_waypoint.nord_world_pose[:, :3] - ee_pose_nord_w[:, :3], dim=-1)
        pos_err_ost = torch.linalg.norm(self.current_waypoint.ost_world_pose[:, :3] - ee_pose_ost_w[:, :3], dim=-1)
        pos_err_west = torch.linalg.norm(self.current_waypoint.west_world_pose[:, :3] - ee_pose_west_w[:, :3], dim=-1)
        pos_err_sud = torch.linalg.norm(self.current_waypoint.sud_world_pose[:, :3] - ee_pose_sud_w[:, :3], dim=-1)

        within_pos = (torch.all(pos_err_nord <= self.position_tol) and
                      torch.all(pos_err_ost <= self.position_tol) and
                      torch.all(pos_err_west <= self.position_tol) and
                      torch.all(pos_err_sud <= self.position_tol))

        if self.controls_orientation:
            quat_err_nord = 1.0 - torch.abs(torch.sum(self.current_waypoint.nord_world_pose[:, 3:] * ee_pose_nord_w[:, 3:], dim=-1))
            quat_err_ost = 1.0 - torch.abs(torch.sum(self.current_waypoint.ost_world_pose[:, 3:] * ee_pose_ost_w[:, 3:], dim=-1))
            quat_err_west = 1.0 - torch.abs(torch.sum(self.current_waypoint.west_world_pose[:, 3:] * ee_pose_west_w[:, 3:], dim=-1))
            quat_err_sud = 1.0 - torch.abs(torch.sum(self.current_waypoint.sud_world_pose[:, 3:] * ee_pose_sud_w[:, 3:], dim=-1))

            within_rot = (torch.all(quat_err_nord <= self.orientation_tol) and
                         torch.all(quat_err_ost <= self.orientation_tol) and
                         torch.all(quat_err_west <= self.orientation_tol) and
                         torch.all(quat_err_sud <= self.orientation_tol))
        else:
            within_rot = True

        if within_pos and within_rot:
            self.hold_counter += 1
        else:
            self.hold_counter = 0

        converged = self.hold_counter >= self.required_hold

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

    def _compose_quad_arm_action(self, nord_joints, nord_gripper, ost_joints, ost_gripper,
                                  west_joints, west_gripper, sud_joints, sud_gripper):
        """Compose 24-DOF quad-arm action tensor."""
        action = torch.zeros(self.env.num_envs, 24, device=self.env.device)
        action[:, :5] = nord_joints[:, :5]
        action[:, 5] = nord_gripper
        action[:, 6:11] = ost_joints[:, :5]
        action[:, 11] = ost_gripper
        action[:, 12:17] = west_joints[:, :5]
        action[:, 17] = west_gripper
        action[:, 18:23] = sud_joints[:, :5]
        action[:, 23] = sud_gripper
        return action


def load_waypoints_from_json(filepath: str, device: str) -> list[WaypointCommand]:
    """Load waypoints from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Waypoint file must contain a JSON array")

    waypoints = []
    for idx, wp in enumerate(data):
        if "nord" not in wp or "ost" not in wp or "west" not in wp or "sud" not in wp:
            raise ValueError(f"Waypoint {idx} missing arm data")

        nord_pose = _parse_waypoint_pose(wp["nord"], device)
        ost_pose = _parse_waypoint_pose(wp["ost"], device)
        west_pose = _parse_waypoint_pose(wp["west"], device)
        sud_pose = _parse_waypoint_pose(wp["sud"], device)

        waypoints.append(WaypointCommand(
            nord_world_pose=nord_pose.unsqueeze(0),
            ost_world_pose=ost_pose.unsqueeze(0),
            west_world_pose=west_pose.unsqueeze(0),
            sud_world_pose=sud_pose.unsqueeze(0),
            nord_gripper=wp["nord"].get("gripper", 0.0),
            ost_gripper=wp["ost"].get("gripper", 0.0),
            west_gripper=wp["west"].get("gripper", 0.0),
            sud_gripper=wp["sud"].get("gripper", 0.0),
            hold_steps=wp.get("hold_steps", 30),
            nord_wrist_flex=wp["nord"].get("wrist_flex", None),
            ost_wrist_flex=wp["ost"].get("wrist_flex", None),
            west_wrist_flex=wp["west"].get("wrist_flex", None),
            sud_wrist_flex=wp["sud"].get("wrist_flex", None),
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

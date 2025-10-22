# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waypoint controller for bi-arm manipulation."""

import torch
import json
from dataclasses import dataclass
from typing import Optional

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat, quat_inv


# Constants
LEFT_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
RIGHT_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
IDENTITY_QUAT = [1.0, 0.0, 0.0, 0.0]


@dataclass
class WaypointCommand:
    """Single waypoint command for bi-arm."""
    left_world_pose: torch.Tensor  # (1, 7) - position + quaternion
    right_world_pose: torch.Tensor  # (1, 7)
    left_gripper: float
    right_gripper: float
    hold_steps: int


class BiArmWaypointController:
    """Controller for executing waypoint sequences on bi-arm robot."""

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        command_type: str = "position",  # "position" or "pose"
        ik_method: str = "dls",
        position_tol: float = 0.01,
        orientation_tol: float = 0.02,
        pose_interp_gain: float = 1.0,
        interp_gain: float = 1.0,
    ):
        """
        Initialize waypoint controller.

        Args:
            env: Environment instance
            command_type: "position" (position-only) or "pose" (position+orientation)
            ik_method: IK solver method ("dls", "pinv", "svd", "trans")
            position_tol: Position convergence tolerance in meters
            orientation_tol: Orientation convergence tolerance (1 - |dot|)
            pose_interp_gain: Interpolation gain for pose (0-1)
            interp_gain: Interpolation gain for joint targets (0-1)
        """
        self.env = env
        self.position_tol = position_tol
        self.orientation_tol = orientation_tol
        self.pose_interp_gain = pose_interp_gain
        self.interp_gain = interp_gain

        # Setup arms
        self.left_arm = env.scene["left_arm"]
        self.right_arm = env.scene["right_arm"]

        self.left_entity = SceneEntityCfg("left_arm", joint_names=LEFT_JOINT_NAMES, body_names=["gripper"])
        self.right_entity = SceneEntityCfg("right_arm", joint_names=RIGHT_JOINT_NAMES, body_names=["gripper"])
        self.left_entity.resolve(env.scene)
        self.right_entity.resolve(env.scene)

        # Setup IK controllers
        ik_cfg = DifferentialIKControllerCfg(command_type=command_type, use_relative_mode=False, ik_method=ik_method)
        self.left_ik = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)
        self.right_ik = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)
        self.controls_orientation = ik_cfg.command_type == "pose"

        # Setup Jacobian indices
        self.left_jacobian_index = self.left_entity.body_ids[0] - 1 if self.left_arm.is_fixed_base else self.left_entity.body_ids[0]
        self.right_jacobian_index = self.right_entity.body_ids[0] - 1 if self.right_arm.is_fixed_base else self.right_entity.body_ids[0]

        self.left_jacobian_joint_ids = self._resolve_joint_indices(self.left_entity.joint_ids, self.left_arm)
        self.right_jacobian_joint_ids = self._resolve_joint_indices(self.right_entity.joint_ids, self.right_arm)

        if not self.left_arm.is_fixed_base:
            self.left_jacobian_joint_ids = [idx + 6 for idx in self.left_jacobian_joint_ids]
        if not self.right_arm.is_fixed_base:
            self.right_jacobian_joint_ids = [idx + 6 for idx in self.right_jacobian_joint_ids]

        # State
        self.commanded_left_world = None
        self.commanded_right_world = None
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
        self.commanded_left_world = None
        self.commanded_right_world = None
        self.current_waypoint = None
        self.hold_counter = 0
        self.required_hold = 0

    def set_waypoint(self, waypoint: WaypointCommand, hold_steps_override: Optional[int] = None):
        """
        Set new target waypoint.

        Args:
            waypoint: Target waypoint command
            hold_steps_override: Override waypoint's hold_steps if provided
        """
        self.current_waypoint = waypoint
        self.hold_counter = 0
        self.required_hold = hold_steps_override if hold_steps_override is not None else waypoint.hold_steps

        # Initialize commanded pose to current EE pose
        ee_pose_left_w = self.left_arm.data.body_pose_w[:, self.left_entity.body_ids[0]]
        ee_pose_right_w = self.right_arm.data.body_pose_w[:, self.right_entity.body_ids[0]]

        if self.commanded_left_world is None:
            self.commanded_left_world = ee_pose_left_w.clone()
            self.commanded_right_world = ee_pose_right_w.clone()
        else:
            # Use current EE pose for smooth transition
            self.commanded_left_world = ee_pose_left_w.clone()
            self.commanded_right_world = ee_pose_right_w.clone()

        self.commanded_left_world[:, 3:] = self._normalize_quat(self.commanded_left_world[:, 3:])
        self.commanded_right_world[:, 3:] = self._normalize_quat(self.commanded_right_world[:, 3:])

        # Reset IK controllers
        self.left_ik.reset()
        self.right_ik.reset()

        # Set initial IK command
        root_pose_left = self.left_arm.data.root_pose_w
        root_pose_right = self.right_arm.data.root_pose_w

        if self.controls_orientation:
            desired_left_base = self._convert_world_to_base(self.commanded_left_world.view(-1), root_pose_left).view(self.env.num_envs, -1)
            desired_right_base = self._convert_world_to_base(self.commanded_right_world.view(-1), root_pose_right).view(self.env.num_envs, -1)
            self.left_ik.set_command(desired_left_base)
            self.right_ik.set_command(desired_right_base)
        else:
            ee_pos_left_b, ee_quat_left_b = subtract_frame_transforms(
                root_pose_left[:, 0:3], root_pose_left[:, 3:7],
                ee_pose_left_w[:, 0:3], ee_pose_left_w[:, 3:7]
            )
            ee_pos_right_b, ee_quat_right_b = subtract_frame_transforms(
                root_pose_right[:, 0:3], root_pose_right[:, 3:7],
                ee_pose_right_w[:, 0:3], ee_pose_right_w[:, 3:7]
            )
            desired_left_pos_base = self._convert_world_to_base(self.commanded_left_world.view(-1), root_pose_left).view(self.env.num_envs, -1)[:, :3]
            desired_right_pos_base = self._convert_world_to_base(self.commanded_right_world.view(-1), root_pose_right).view(self.env.num_envs, -1)[:, :3]
            self.left_ik.set_command(desired_left_pos_base, ee_quat=ee_quat_left_b)
            self.right_ik.set_command(desired_right_pos_base, ee_quat=ee_quat_right_b)

    def step(self) -> tuple[torch.Tensor, bool]:
        """
        Execute one control step towards current waypoint.

        Returns:
            action: (num_envs, 12) joint action tensor
            converged: True if waypoint reached and held
        """
        if self.current_waypoint is None:
            raise RuntimeError("No waypoint set. Call set_waypoint() first.")

        # Get current state
        root_pose_left = self.left_arm.data.root_pose_w
        root_pose_right = self.right_arm.data.root_pose_w
        ee_pose_left_w = self.left_arm.data.body_pose_w[:, self.left_entity.body_ids[0]]
        ee_pose_right_w = self.right_arm.data.body_pose_w[:, self.right_entity.body_ids[0]]

        # Get Jacobians
        jacobian_left_w = self.left_arm.root_physx_view.get_jacobians()[:, self.left_jacobian_index, :, self.left_jacobian_joint_ids]
        jacobian_right_w = self.right_arm.root_physx_view.get_jacobians()[:, self.right_jacobian_index, :, self.right_jacobian_joint_ids]

        # Transform to base frame
        ee_pos_left_b, ee_quat_left_b = subtract_frame_transforms(
            root_pose_left[:, 0:3], root_pose_left[:, 3:7],
            ee_pose_left_w[:, 0:3], ee_pose_left_w[:, 3:7]
        )
        ee_pos_right_b, ee_quat_right_b = subtract_frame_transforms(
            root_pose_right[:, 0:3], root_pose_right[:, 3:7],
            ee_pose_right_w[:, 0:3], ee_pose_right_w[:, 3:7]
        )

        # Transform Jacobians
        rot_left = matrix_from_quat(quat_inv(root_pose_left[:, 3:7]))
        rot_right = matrix_from_quat(quat_inv(root_pose_right[:, 3:7]))
        jacobian_left = jacobian_left_w.clone()
        jacobian_left[:, :3, :] = torch.bmm(rot_left, jacobian_left_w[:, :3, :])
        jacobian_left[:, 3:, :] = torch.bmm(rot_left, jacobian_left_w[:, 3:, :])
        jacobian_right = jacobian_right_w.clone()
        jacobian_right[:, :3, :] = torch.bmm(rot_right, jacobian_right_w[:, :3, :])
        jacobian_right[:, 3:, :] = torch.bmm(rot_right, jacobian_right_w[:, 3:, :])

        # Interpolate towards target
        if self.controls_orientation:
            self.commanded_left_world = torch.lerp(self.commanded_left_world, self.current_waypoint.left_world_pose, self.pose_interp_gain)
            self.commanded_right_world = torch.lerp(self.commanded_right_world, self.current_waypoint.right_world_pose, self.pose_interp_gain)
            self.commanded_left_world[:, 3:] = self._normalize_quat(self.commanded_left_world[:, 3:])
            self.commanded_right_world[:, 3:] = self._normalize_quat(self.commanded_right_world[:, 3:])
            desired_left_base = self._convert_world_to_base(self.commanded_left_world.view(-1), root_pose_left).view(self.env.num_envs, -1)
            desired_right_base = self._convert_world_to_base(self.commanded_right_world.view(-1), root_pose_right).view(self.env.num_envs, -1)
            self.left_ik.set_command(desired_left_base)
            self.right_ik.set_command(desired_right_base)
        else:
            commanded_left_pos = torch.lerp(self.commanded_left_world[:, :3], self.current_waypoint.left_world_pose[:, :3], self.pose_interp_gain)
            commanded_right_pos = torch.lerp(self.commanded_right_world[:, :3], self.current_waypoint.right_world_pose[:, :3], self.pose_interp_gain)
            self.commanded_left_world[:, :3] = commanded_left_pos
            self.commanded_right_world[:, :3] = commanded_right_pos
            desired_left_pos_base = self._convert_world_to_base(self.commanded_left_world.view(-1), root_pose_left).view(self.env.num_envs, -1)[:, :3]
            desired_right_pos_base = self._convert_world_to_base(self.commanded_right_world.view(-1), root_pose_right).view(self.env.num_envs, -1)[:, :3]
            self.left_ik.set_command(desired_left_pos_base, ee_quat=ee_quat_left_b)
            self.right_ik.set_command(desired_right_pos_base, ee_quat=ee_quat_right_b)

        # Compute IK
        joint_pos_left = self.left_arm.data.joint_pos[:, self.left_entity.joint_ids]
        joint_pos_right = self.right_arm.data.joint_pos[:, self.right_entity.joint_ids]

        if self.controls_orientation:
            joint_targets_left = self.left_ik.compute(ee_pos_left_b, ee_quat_left_b, jacobian_left, joint_pos_left)
            joint_targets_right = self.right_ik.compute(ee_pos_right_b, ee_quat_right_b, jacobian_right, joint_pos_right)
        else:
            joint_targets_left = self.left_ik.compute(ee_pos_left_b, ee_quat_left_b, jacobian_left[:, :3, :], joint_pos_left)
            joint_targets_right = self.right_ik.compute(ee_pos_right_b, ee_quat_right_b, jacobian_right[:, :3, :], joint_pos_right)

        # Apply interpolation gain
        if self.interp_gain < 1.0:
            joint_targets_left = joint_pos_left + self.interp_gain * (joint_targets_left - joint_pos_left)
            joint_targets_right = joint_pos_right + self.interp_gain * (joint_targets_right - joint_pos_right)

        # Compose action
        action = self._compose_bi_arm_action(
            joint_targets_left,
            self.current_waypoint.left_gripper,
            joint_targets_right,
            self.current_waypoint.right_gripper
        )

        # Check convergence
        pos_err_left = torch.linalg.norm(self.current_waypoint.left_world_pose[:, :3] - ee_pose_left_w[:, :3], dim=-1)
        pos_err_right = torch.linalg.norm(self.current_waypoint.right_world_pose[:, :3] - ee_pose_right_w[:, :3], dim=-1)
        within_pos = torch.all(pos_err_left <= self.position_tol) and torch.all(pos_err_right <= self.position_tol)

        if self.controls_orientation:
            quat_err_left = 1.0 - torch.abs(torch.sum(self.current_waypoint.left_world_pose[:, 3:] * ee_pose_left_w[:, 3:], dim=-1))
            quat_err_right = 1.0 - torch.abs(torch.sum(self.current_waypoint.right_world_pose[:, 3:] * ee_pose_right_w[:, 3:], dim=-1))
            within_rot = torch.all(quat_err_left <= self.orientation_tol) and torch.all(quat_err_right <= self.orientation_tol)
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
        # Ensure proper shapes: world_pose is (7,), root_pose is (1, 7)
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

    def _compose_bi_arm_action(self, left_joints, left_gripper, right_joints, right_gripper):
        """Compose 12-DOF bi-arm action tensor."""
        action = torch.zeros(self.env.num_envs, 12, device=self.env.device)
        action[:, :5] = left_joints[:, :5]
        action[:, 5] = left_gripper
        action[:, 6:11] = right_joints[:, :5]
        action[:, 11] = right_gripper
        return action


def load_waypoints_from_json(filepath: str, device: str) -> list[WaypointCommand]:
    """Load waypoints from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Waypoint file must contain a JSON array")

    waypoints = []
    for idx, wp in enumerate(data):
        if "left" not in wp or "right" not in wp:
            raise ValueError(f"Waypoint {idx} missing 'left' or 'right' arm data")

        left_pose = _parse_waypoint_pose(wp["left"], device)
        right_pose = _parse_waypoint_pose(wp["right"], device)

        waypoints.append(WaypointCommand(
            left_world_pose=left_pose.unsqueeze(0),
            right_world_pose=right_pose.unsqueeze(0),
            left_gripper=wp["left"].get("gripper", 0.0),
            right_gripper=wp["right"].get("gripper", 0.0),
            hold_steps=wp.get("hold_steps", 30),
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

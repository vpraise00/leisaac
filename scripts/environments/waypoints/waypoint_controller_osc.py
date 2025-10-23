# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waypoint controller for bi-arm manipulation using OperationalSpaceController."""

import torch
import json
from dataclasses import dataclass
from typing import Optional

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
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
    left_wrist_flex: Optional[float] = None  # Optional wrist_flex angle in radians
    right_wrist_flex: Optional[float] = None  # Optional wrist_flex angle in radians


class BiArmWaypointControllerOSC:
    """Controller for executing waypoint sequences on bi-arm robot using OperationalSpaceController."""

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        command_type: str = "pose_abs",  # "pose_abs" or "pose_rel"
        position_tol: float = 0.01,
        orientation_tol: float = 0.02,
        pose_interp_gain: float = 1.0,
        motion_stiffness: float = 150.0,
        motion_damping_ratio: float = 1.0,
        force_wrist_down: bool = True,
        wrist_flex_angle: float = 1.57,  # ~90 degrees in radians
    ):
        """
        Initialize waypoint controller with OperationalSpaceController.

        Args:
            env: Environment instance
            command_type: "pose_abs" (absolute pose) or "pose_rel" (relative pose)
            position_tol: Position convergence tolerance in meters
            orientation_tol: Orientation convergence tolerance (1 - |dot|)
            pose_interp_gain: Interpolation gain for pose (0-1)
            motion_stiffness: Stiffness for impedance control (higher = stiffer)
            motion_damping_ratio: Damping ratio for velocity control
            force_wrist_down: Force wrist_flex joint to point downward
            wrist_flex_angle: Target angle for wrist_flex joint in radians (default: 1.57 = 90 deg)
        """
        self.env = env
        self.position_tol = position_tol
        self.orientation_tol = orientation_tol
        self.pose_interp_gain = pose_interp_gain
        self.force_wrist_down = force_wrist_down
        self.wrist_flex_angle = wrist_flex_angle

        # Setup arms
        self.left_arm = env.scene["left_arm"]
        self.right_arm = env.scene["right_arm"]

        self.left_entity = SceneEntityCfg("left_arm", joint_names=LEFT_JOINT_NAMES, body_names=["gripper"])
        self.right_entity = SceneEntityCfg("right_arm", joint_names=RIGHT_JOINT_NAMES, body_names=["gripper"])
        self.left_entity.resolve(env.scene)
        self.right_entity.resolve(env.scene)

        # Setup OperationalSpace controllers
        osc_cfg = OperationalSpaceControllerCfg(
            target_types=[command_type],
            impedance_mode="fixed",
            inertial_dynamics_decoupling=False,  # Disable to avoid needing mass matrix
            gravity_compensation=False,  # Disable to simplify (can enable if gravity tensor available)
            motion_control_axes_task=[1, 1, 1, 1, 1, 1],  # Control all 6 DOF
            motion_stiffness_task=motion_stiffness,
            motion_damping_ratio_task=motion_damping_ratio,
            nullspace_control="none",  # SO-101 is 6-DOF, not redundant
        )
        self.left_osc = OperationalSpaceController(osc_cfg, num_envs=env.num_envs, device=env.device)
        self.right_osc = OperationalSpaceController(osc_cfg, num_envs=env.num_envs, device=env.device)
        self.command_type = command_type

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
        self.left_osc.reset()
        self.right_osc.reset()

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

        # Set initial OSC command
        root_pose_left = self.left_arm.data.root_pose_w
        root_pose_right = self.right_arm.data.root_pose_w

        desired_left_base = self._convert_world_to_base(self.commanded_left_world, root_pose_left)
        desired_right_base = self._convert_world_to_base(self.commanded_right_world, root_pose_right)

        # Get current EE pose in base frame
        ee_pose_left_w_current = self.left_arm.data.body_pose_w[:, self.left_entity.body_ids[0]]
        ee_pose_right_w_current = self.right_arm.data.body_pose_w[:, self.right_entity.body_ids[0]]
        current_left_base = self._convert_world_to_base(ee_pose_left_w_current, root_pose_left)
        current_right_base = self._convert_world_to_base(ee_pose_right_w_current, root_pose_right)

        self.left_osc.set_command(desired_left_base, current_ee_pose_b=current_left_base)
        self.right_osc.set_command(desired_right_base, current_ee_pose_b=current_right_base)

    def step(self) -> tuple[torch.Tensor, bool]:
        """
        Execute one control step towards current waypoint.

        Returns:
            action: (num_envs, 12) joint action tensor (6 joints per arm)
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

        ee_pose_left_b = torch.cat([ee_pos_left_b, ee_quat_left_b], dim=-1)
        ee_pose_right_b = torch.cat([ee_pos_right_b, ee_quat_right_b], dim=-1)

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
        self.commanded_left_world = torch.lerp(self.commanded_left_world, self.current_waypoint.left_world_pose, self.pose_interp_gain)
        self.commanded_right_world = torch.lerp(self.commanded_right_world, self.current_waypoint.right_world_pose, self.pose_interp_gain)
        self.commanded_left_world[:, 3:] = self._normalize_quat(self.commanded_left_world[:, 3:])
        self.commanded_right_world[:, 3:] = self._normalize_quat(self.commanded_right_world[:, 3:])

        desired_left_base = self._convert_world_to_base(self.commanded_left_world, root_pose_left)
        desired_right_base = self._convert_world_to_base(self.commanded_right_world, root_pose_right)

        self.left_osc.set_command(desired_left_base, current_ee_pose_b=ee_pose_left_b)
        self.right_osc.set_command(desired_right_base, current_ee_pose_b=ee_pose_right_b)

        # Get joint states
        joint_pos_left = self.left_arm.data.joint_pos[:, self.left_entity.joint_ids]
        joint_pos_right = self.right_arm.data.joint_pos[:, self.right_entity.joint_ids]
        joint_vel_left = self.left_arm.data.joint_vel[:, self.left_entity.joint_ids]
        joint_vel_right = self.right_arm.data.joint_vel[:, self.right_entity.joint_ids]

        # Compute end-effector velocity in base frame using Jacobian
        ee_vel_left_b = torch.bmm(jacobian_left, joint_vel_left.unsqueeze(-1)).squeeze(-1)
        ee_vel_right_b = torch.bmm(jacobian_right, joint_vel_right.unsqueeze(-1)).squeeze(-1)

        # Compute OSC (returns joint efforts, but we'll use them as position targets with scaling)
        # Note: OSC returns efforts, so we need to integrate them or use as velocity commands
        # For simplicity, we'll compute target joint positions by inverting the dynamics
        efforts_left = self.left_osc.compute(
            jacobian_b=jacobian_left,
            current_ee_pose_b=ee_pose_left_b,
            current_ee_vel_b=ee_vel_left_b,
            current_joint_pos=joint_pos_left,
            current_joint_vel=joint_vel_left,
        )
        efforts_right = self.right_osc.compute(
            jacobian_b=jacobian_right,
            current_ee_pose_b=ee_pose_right_b,
            current_ee_vel_b=ee_vel_right_b,
            current_joint_pos=joint_pos_right,
            current_joint_vel=joint_vel_right,
        )

        # Convert efforts to position targets (simple proportional conversion)
        # This is a simplified approach - in practice you'd use the efforts directly
        # or integrate them properly with the dynamics model
        effort_to_pos_gain = 0.01
        joint_targets_left = joint_pos_left + efforts_left * effort_to_pos_gain
        joint_targets_right = joint_pos_right + efforts_right * effort_to_pos_gain

        # Force wrist_flex to point downward if enabled
        wrist_flex_idx = 3

        if self.current_waypoint.left_wrist_flex is not None:
            joint_targets_left[:, wrist_flex_idx] = self.current_waypoint.left_wrist_flex
        elif self.force_wrist_down:
            joint_targets_left[:, wrist_flex_idx] = self.wrist_flex_angle

        if self.current_waypoint.right_wrist_flex is not None:
            joint_targets_right[:, wrist_flex_idx] = self.current_waypoint.right_wrist_flex
        elif self.force_wrist_down:
            joint_targets_right[:, wrist_flex_idx] = self.wrist_flex_angle

        # Set gripper targets
        joint_targets_left[:, -1] = self.current_waypoint.left_gripper
        joint_targets_right[:, -1] = self.current_waypoint.right_gripper

        # Check convergence
        position_converged = self._check_position_convergence(ee_pose_left_w, ee_pose_right_w)

        if position_converged:
            self.hold_counter += 1
        else:
            self.hold_counter = 0

        converged = self.hold_counter >= self.required_hold

        # Combine actions
        action = torch.cat([joint_targets_left, joint_targets_right], dim=-1)
        return action, converged

    def _check_position_convergence(self, ee_pose_left_w, ee_pose_right_w):
        """Check if end-effector positions have converged to target."""
        left_pos_error = torch.norm(ee_pose_left_w[:, :3] - self.current_waypoint.left_world_pose[:, :3], dim=-1)
        right_pos_error = torch.norm(ee_pose_right_w[:, :3] - self.current_waypoint.right_world_pose[:, :3], dim=-1)

        left_converged = (left_pos_error < self.position_tol).all()
        right_converged = (right_pos_error < self.position_tol).all()

        return left_converged and right_converged

    def _convert_world_to_base(self, world_pose, root_pose):
        """Convert world-frame pose to base-frame pose."""
        # Ensure world_pose has batch dimension
        if world_pose.ndim == 1:
            world_pose = world_pose.unsqueeze(0)

        if world_pose.shape[-1] == 7:
            pos_b, quat_b = subtract_frame_transforms(
                root_pose[:, 0:3], root_pose[:, 3:7],
                world_pose[:, 0:3], world_pose[:, 3:7]
            )
            return torch.cat([pos_b, quat_b], dim=-1)
        else:
            pos_b, _ = subtract_frame_transforms(
                root_pose[:, 0:3], root_pose[:, 3:7],
                world_pose[:, 0:3], torch.tensor([[1, 0, 0, 0]], device=world_pose.device, dtype=torch.float32).expand(world_pose.shape[0], 4)
            )
            return pos_b

    def _normalize_quat(self, quat):
        """Normalize quaternion."""
        return quat / torch.norm(quat, dim=-1, keepdim=True)


def load_waypoints_from_json(json_path: str, device: str) -> list[WaypointCommand]:
    """
    Load waypoints from JSON file.

    Args:
        json_path: Path to waypoints JSON file
        device: PyTorch device string

    Returns:
        List of WaypointCommand objects
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    waypoints = []
    for wp in data:
        # Parse left arm
        left_pos = torch.tensor(wp["left"]["position"], dtype=torch.float32, device=device)
        left_quat = torch.tensor(wp["left"].get("orientation", IDENTITY_QUAT), dtype=torch.float32, device=device)
        left_pose = torch.cat([left_pos, left_quat]).unsqueeze(0)  # Shape: (1, 7)

        # Parse right arm
        right_pos = torch.tensor(wp["right"]["position"], dtype=torch.float32, device=device)
        right_quat = torch.tensor(wp["right"].get("orientation", IDENTITY_QUAT), dtype=torch.float32, device=device)
        right_pose = torch.cat([right_pos, right_quat]).unsqueeze(0)  # Shape: (1, 7)

        waypoints.append(WaypointCommand(
            left_world_pose=left_pose,  # Keep as (1, 7)
            right_world_pose=right_pose,  # Keep as (1, 7)
            left_gripper=wp["left"].get("gripper", 0.0),
            right_gripper=wp["right"].get("gripper", 0.0),
            hold_steps=wp.get("hold_steps", 30),
            left_wrist_flex=wp["left"].get("wrist_flex", None),
            right_wrist_flex=wp["right"].get("wrist_flex", None),
        ))

    return waypoints

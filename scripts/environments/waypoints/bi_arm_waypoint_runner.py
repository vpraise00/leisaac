# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drive the bi-arm SO101 environment through a list of end-effector waypoints.

This helper runs entirely locally (no policy server) and uses the differential IK
controller provided by IsaacLab to translate target gripper poses into joint-space
commands for both follower arms. It is intended as a lightweight starting point
for generating scripted demonstrations or trajectory data when teleoperation
hardware is not available.

Waypoints can be provided through a JSON file or fall back to a built-in toy
sequence. Each waypoint contains a desired pose for the left and right gripper
expressed in world coordinates (XYZ in meters, orientation as ``[w, x, y, z]``).
Gripper aperture commands are optional and default to constant values.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import torch
import time

from isaaclab.app import AppLauncher

if TYPE_CHECKING:
    from leisaac.assets.robots.lerobot import SO101_FOLLOWER_USD_JOINT_LIMLITS

LEISAAC_IMPORTED = False


# Use "spawn" start method for compatibility with Isaac Sim.
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)


DEFAULT_WAYPOINTS: list[dict[str, Any]] = [
    {
        "relative": True,
        "left": {
            "offset": [0.0, -0.20, 0.22],
            "orientation_offset": [1.0, 0.0, 0.0, 0.0],
            "gripper": 0.2,
        },
        "right": {
            "offset": [0.0, -0.20, 0.22],
            "orientation_offset": [1.0, 0.0, 0.0, 0.0],
            "gripper": 0.2,
        },
        "hold_steps": 60,
    },
    {
        "relative": True,
        "left": {
            "offset": [0.0, -0.18, -0.20],
            "orientation_offset": [1.0, 0.0, 0.0, 0.0],
            "gripper": 0.2,
        },
        "right": {
            "offset": [0.0, -0.18, -0.20],
            "orientation_offset": [1.0, 0.0, 0.0, 0.0],
            "gripper": 0.2,
        },
        "hold_steps": 80,
    },
    {
        "relative": True,
        "left": {
            "offset": [0.0, 0.0, 0.0],
            "orientation_offset": [1.0, 0.0, 0.0, 0.0],
            "gripper": 0.6,
        },
        "right": {
            "offset": [0.0, 0.0, 0.0],
            "orientation_offset": [1.0, 0.0, 0.0, 0.0],
            "gripper": 0.6,
        },
        "hold_steps": 40,
    },
    {
        "relative": True,
        "left": {
            "offset": [0.0, 0.0, 0.0],
            "orientation_offset": [0.0, 0.0, 0.0, 1.0],
            "gripper": 0.0,
        },
        "right": {
            "offset": [0.0, 0.0, 0.0],
            "orientation_offset": [0.0, 0.0, 0.0, 1.0],
            "gripper": 0.0,
        },
        "hold_steps": 40,
    },
    {
        "relative": True,
        "left": {
            "offset": [0.0, 0.15, 0.18],
            "orientation_offset": [0.0, 0.0, 0.0, 1.0],
            "gripper": 0.0,
        },
        "right": {
            "offset": [0.0, 0.15, 0.18],
            "orientation_offset": [0.0, 0.0, 0.0, 1.0],
            "gripper": 0.0,
        },
        "hold_steps": 50,
    },
    {
        "relative": True,
        "left": {
            "offset": [0.0, 0.35, -0.09],
            "orientation_offset": [0.0, 0.0, 0.0, 1.0],
            "gripper": 0.0,
        },
        "right": {
            "offset": [0.0, 0.35, -0.09],
            "orientation_offset": [0.0, 0.0, 0.0, 1.0],
            "gripper": 0.0,
        },
        "hold_steps": 50,
    },
]

IDENTITY_QUAT = [1.0, 0.0, 0.0, 0.0]
LEFT_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
RIGHT_JOINT_NAMES = LEFT_JOINT_NAMES
SUBTRACT_FRAME_TRANSFORMS = None
MAX_X_OFFSET = 0.05


def _build_joint_limit_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert USD joint limits (degrees) into radian tensors on the target device."""
    from leisaac.assets.robots.lerobot import SO101_FOLLOWER_USD_JOINT_LIMLITS

    arm_limits = torch.tensor(
        [
            [math.radians(SO101_FOLLOWER_USD_JOINT_LIMLITS[name][0]), math.radians(SO101_FOLLOWER_USD_JOINT_LIMLITS[name][1])]
            for name in LEFT_JOINT_NAMES
        ],
        dtype=torch.float32,
        device=device,
    )
    gripper_limits = torch.tensor(
        [
            math.radians(SO101_FOLLOWER_USD_JOINT_LIMLITS["gripper"][0]),
            math.radians(SO101_FOLLOWER_USD_JOINT_LIMLITS["gripper"][1]),
        ],
        dtype=torch.float32,
        device=device,
    )
    return arm_limits, gripper_limits


def _compose_bi_arm_action(
    left_arm_targets: torch.Tensor,
    left_gripper_targets: torch.Tensor,
    right_arm_targets: torch.Tensor,
    right_gripper_targets: torch.Tensor,
    arm_limits: torch.Tensor,
    gripper_limits: torch.Tensor,
) -> torch.Tensor:
    """Pack per-arm joint targets into the teleop-style flat tensor expected by the action manager."""
    left_arm_cmd = torch.clamp(left_arm_targets, arm_limits[:, 0], arm_limits[:, 1])
    right_arm_cmd = torch.clamp(right_arm_targets, arm_limits[:, 0], arm_limits[:, 1])
    left_grip_cmd = torch.clamp(left_gripper_targets, gripper_limits[0], gripper_limits[1])
    right_grip_cmd = torch.clamp(right_gripper_targets, gripper_limits[0], gripper_limits[1])
    return torch.cat([left_arm_cmd, left_grip_cmd, right_arm_cmd, right_grip_cmd], dim=-1)


@dataclass
class WaypointCommand:
    """Prepared command buffers for both arms."""

    left_pose: torch.Tensor  # shape: (N, 7)
    right_pose: torch.Tensor  # shape: (N, 7)
    left_world_pose: torch.Tensor  # shape: (N, 7)
    right_world_pose: torch.Tensor  # shape: (N, 7)
    left_gripper: torch.Tensor  # shape: (N, 1)
    right_gripper: torch.Tensor  # shape: (N, 1)
    hold_steps: int


class RateLimiter:
    """Simple helper to throttle simulation stepping."""

    def __init__(self, hz: float) -> None:
        self._sleep_duration = 0.0 if hz <= 0.0 else 1.0 / hz
        self._render_period = 0.0166 if hz > 0.0 else 0.0
        self._last_time = time.time()

    def reset(self) -> None:
        self._last_time = time.time()

    def sleep(self, env) -> None:
        if self._sleep_duration <= 0.0:
            return
        next_wakeup = self._last_time + self._sleep_duration
        while time.time() < next_wakeup:
            env.render()
            time.sleep(min(self._render_period, max(next_wakeup - time.time(), 0.0)))
        self._last_time = next_wakeup


def parse_waypoints(waypoint_file: Path | None) -> list[dict[str, Any]]:
    if waypoint_file is None:
        return DEFAULT_WAYPOINTS
    data = json.loads(Path(waypoint_file).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Waypoint file must contain a list of waypoint objects.")
    if not data:
        raise ValueError("Waypoint list is empty.")
    return data


def _tensorize_pose(
    pose: dict[str, Iterable[float]],
    device: torch.device,
    default_position: Iterable[float] | torch.Tensor | None = None,
    default_orientation: Iterable[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    if "position" in pose:
        position = torch.as_tensor(pose["position"], dtype=torch.float32, device=device)
    elif default_position is not None:
        position = torch.as_tensor(default_position, dtype=torch.float32, device=device)
    else:
        position = torch.zeros(3, dtype=torch.float32, device=device)
    if position.shape != (3,):
        raise ValueError(f"Position must be a 3-vector. Got shape {tuple(position.shape)}")

    if "orientation" in pose:
        orientation = torch.as_tensor(pose["orientation"], dtype=torch.float32, device=device)
    elif default_orientation is not None:
        orientation = torch.as_tensor(default_orientation, dtype=torch.float32, device=device)
    else:
        orientation = torch.tensor(IDENTITY_QUAT, dtype=torch.float32, device=device)
    if orientation.shape != (4,):
        raise ValueError(f"Orientation must be a quaternion [w, x, y, z]. Got shape {tuple(orientation.shape)}")
    orientation = _normalize_quat(orientation.unsqueeze(0)).squeeze(0)
    return torch.cat([position, orientation], dim=0)


def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1e-9)


def _convert_world_to_base(
    target_pose_w: torch.Tensor,
    root_pose_w: torch.Tensor,
) -> torch.Tensor:
    if SUBTRACT_FRAME_TRANSFORMS is None:
        raise RuntimeError("subtract_frame_transforms is not initialized. Call after Isaac Sim imports.")
    pos_w = target_pose_w[:3].unsqueeze(0)
    quat_w = target_pose_w[3:].unsqueeze(0)
    base_pos = root_pose_w[:, 0:3]
    base_quat = root_pose_w[:, 3:7]
    pos_b, quat_b = SUBTRACT_FRAME_TRANSFORMS(base_pos, base_quat, pos_w, quat_w)
    return torch.cat([pos_b, quat_b], dim=-1)


def _apply_relative_offset(
    base_pose: torch.Tensor,
    offset: Iterable[float],
    orientation_offset: Iterable[float],
    device: torch.device,
) -> torch.Tensor:
    offset = torch.tensor(offset, dtype=torch.float32, device=device)
    offset[0] = offset[0].clamp(-MAX_X_OFFSET, MAX_X_OFFSET)
    if offset.shape != (3,):
        raise ValueError(f"Offset must be a 3-vector. Got shape {tuple(offset.shape)}")
    orientation_offset = torch.tensor(orientation_offset, dtype=torch.float32, device=device)
    if orientation_offset.shape != (4,):
        raise ValueError(
            f"Orientation offset must be a quaternion [w, x, y, z]. Got shape {tuple(orientation_offset.shape)}"
        )
    orientation_offset = _normalize_quat(orientation_offset.unsqueeze(0)).squeeze(0)

    pos = base_pose[:3] + offset
    quat = _quat_multiply(base_pose[3:], orientation_offset)
    return torch.cat([pos, quat], dim=0)


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion multiply (w, x, y, z convention)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    quat = torch.stack([w, x, y, z], dim=0)
    return _normalize_quat(quat.unsqueeze(0)).squeeze(0)


def prepare_waypoint(
    waypoint: dict[str, Any],
    left_root_pose: torch.Tensor,
    right_root_pose: torch.Tensor,
    current_left_world: torch.Tensor,
    current_right_world: torch.Tensor,
    device: torch.device,
    default_left_gripper: float,
    default_right_gripper: float,
    num_envs: int,
) -> tuple[WaypointCommand, torch.Tensor, torch.Tensor]:
    left_entry = waypoint.get("left", {})
    right_entry = waypoint.get("right", {})
    relative = bool(waypoint.get("relative", False))

    if relative:
        left_offset = left_entry.get("offset", [0.0, 0.0, 0.0])
        right_offset = right_entry.get("offset", [0.0, 0.0, 0.0])
        left_orientation_offset = left_entry.get("orientation_offset", IDENTITY_QUAT)
        right_orientation_offset = right_entry.get("orientation_offset", IDENTITY_QUAT)

        left_pose_world = _apply_relative_offset(
            current_left_world, left_offset, left_orientation_offset, device
        )
        right_pose_world = _apply_relative_offset(
            current_right_world, right_offset, right_orientation_offset, device
        )
    else:
        left_pose_world = _tensorize_pose(
            left_entry,
            device,
            default_position=current_left_world[:3],
            default_orientation=current_left_world[3:],
        )
        right_pose_world = _tensorize_pose(
            right_entry,
            device,
            default_position=current_right_world[:3],
            default_orientation=current_right_world[3:],
        )

    left_pose_b = _convert_world_to_base(left_pose_world, left_root_pose).view(num_envs, -1)
    right_pose_b = _convert_world_to_base(right_pose_world, right_root_pose).view(num_envs, -1)

    left_gripper_val = left_entry.get(
        "gripper",
        waypoint.get("left_gripper", waypoint.get("gripper", default_left_gripper)),
    )
    right_gripper_val = right_entry.get(
        "gripper",
        waypoint.get("right_gripper", waypoint.get("gripper", default_right_gripper)),
    )
    left_gripper = torch.full((num_envs, 1), float(left_gripper_val), device=device, dtype=torch.float32)
    right_gripper = torch.full((num_envs, 1), float(right_gripper_val), device=device, dtype=torch.float32)

    hold_steps = int(waypoint.get("hold_steps", 0))

    command = WaypointCommand(
        left_pose=left_pose_b,
        right_pose=right_pose_b,
        left_world_pose=left_pose_world.view(1, -1),
        right_world_pose=right_pose_world.view(1, -1),
        left_gripper=left_gripper,
        right_gripper=right_gripper,
        hold_steps=hold_steps,
    )

    return command, left_pose_world.view(-1), right_pose_world.view(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bi-arm waypoint runner using differential IK.")
    parser.add_argument("--task", type=str, default="LeIsaac-SO101-CleanToyTable-BiArm-v0", help="Registered task name.")
    parser.add_argument("--waypoint_file", type=Path, default=None, help="Optional JSON file with waypoints.")
    parser.add_argument("--step_hz", type=float, default=60.0, help="Control rate for stepping the simulation.")
    parser.add_argument("--position_tol", type=float, default=0.01, help="Position convergence tolerance in meters.")
    parser.add_argument("--orientation_tol", type=float, default=0.02, help="Quaternion convergence tolerance (1 - |dot|).")
    parser.add_argument("--hold_steps", type=int, default=15, help="Minimum consecutive steps to satisfy tolerances before switching waypoints.")
    parser.add_argument("--max_steps", type=int, default=15000, help="Hard limit on simulation steps before exit.")
    parser.add_argument("--left_gripper", type=float, default=0.0, help="Fallback gripper command for the left arm.")
    parser.add_argument("--right_gripper", type=float, default=0.0, help="Fallback gripper command for the right arm.")
    parser.add_argument(
        "--interp_gain",
        type=float,
        default=0.2,
        help="Blend factor in [0, 1] for moving joints toward the IK solution each step (smaller is slower).",
    )
    parser.add_argument(
        "--pose_interp_gain",
        type=float,
        default=0.2,
        help="Blend factor in [0, 1] for dragging the desired pose toward the waypoint each step.",
    )
    parser.add_argument("--record", action="store_true", help="Keep recorder manager enabled for dataset capture.")
    parser.add_argument(
        "--logging_pos",
        action="store_true",
        help="Print the left/right gripper world positions every 2 seconds during execution.",
    )
    parser.add_argument(
        "--debug_ik",
        action="store_true",
        help="Print detailed IK debug information (first few steps).",
    )
    parser.add_argument(
        "--stay_alive",
        action="store_true",
        help="Keep the simulator running after all waypoints (or max steps) are completed.",
    )

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(vars(args_cli))
    simulation_app = app_launcher.app

    global LEISAAC_IMPORTED
    if not LEISAAC_IMPORTED:
        import leisaac  # noqa: F401  (register environments)
        LEISAAC_IMPORTED = True

    import gymnasium as gym
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms
    from isaaclab_tasks.utils import parse_env_cfg

    global SUBTRACT_FRAME_TRANSFORMS
    SUBTRACT_FRAME_TRANSFORMS = subtract_frame_transforms

    # Configure environment
    device_name = getattr(args_cli, "device", "cuda")
    env_cfg = parse_env_cfg(args_cli.task, device=device_name, num_envs=1)
    env_cfg.use_teleop_device("bi-so101leader")
    if not args_cli.record:
        env_cfg.recorders = None
    env_cfg.seed = 0

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    arm_limits, gripper_limits = _build_joint_limit_tensors(env.device)
    interp_gain = max(0.0, min(1.0, args_cli.interp_gain))
    pose_gain = max(0.0, min(1.0, args_cli.pose_interp_gain))

    rate_limiter = RateLimiter(args_cli.step_hz)

    # Resolve scene entities
    left_arm = env.scene["left_arm"]
    right_arm = env.scene["right_arm"]

    left_entity = SceneEntityCfg("left_arm", joint_names=LEFT_JOINT_NAMES, body_names=["gripper"])
    right_entity = SceneEntityCfg("right_arm", joint_names=RIGHT_JOINT_NAMES, body_names=["gripper"])
    left_entity.resolve(env.scene)
    right_entity.resolve(env.scene)

    ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    left_ik = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)
    right_ik = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)
    controls_orientation = ik_cfg.command_type == "pose"

    _obs, _ = env.reset()
    for _ in range(5):
        simulation_app.update()
    rate_limiter.reset()

    # Capture root poses for coordinate transforms
    left_root_pose = left_arm.data.root_pose_w.clone()
    right_root_pose = right_arm.data.root_pose_w.clone()
    initial_left_world = left_arm.data.body_pose_w[:, left_entity.body_ids[0]].clone().view(-1)
    initial_right_world = right_arm.data.body_pose_w[:, right_entity.body_ids[0]].clone().view(-1)

    commanded_left_world = initial_left_world.view(1, -1).clone()
    commanded_right_world = initial_right_world.view(1, -1).clone()
    commanded_left_world[:, 3:] = _normalize_quat(commanded_left_world[:, 3:])
    commanded_right_world[:, 3:] = _normalize_quat(commanded_right_world[:, 3:])

    raw_waypoints = parse_waypoints(args_cli.waypoint_file)
    prepared_waypoints: list[WaypointCommand] = []
    current_left_world = initial_left_world
    current_right_world = initial_right_world
    for wp in raw_waypoints:
        command, current_left_world, current_right_world = prepare_waypoint(
            waypoint=wp,
            left_root_pose=left_root_pose,
            right_root_pose=right_root_pose,
            current_left_world=current_left_world,
            current_right_world=current_right_world,
            device=env.device,
            default_left_gripper=args_cli.left_gripper,
            default_right_gripper=args_cli.right_gripper,
            num_envs=env.num_envs,
        )
        prepared_waypoints.append(command)

    print("[WaypointRunner] Waypoint summary:")
    for idx, wp in enumerate(prepared_waypoints, start=1):
        left_pos = [f"{x:.3f}" for x in wp.left_world_pose[0, :3].tolist()]
        right_pos = [f"{x:.3f}" for x in wp.right_world_pose[0, :3].tolist()]
        left_grip = wp.left_gripper[0].item()
        right_grip = wp.right_gripper[0].item()
        print(
            f"  {idx}: "
            f"left=[{','.join(left_pos)}], right=[{','.join(right_pos)}], "
            f"grip_L={left_grip:.2f}, grip_R={right_grip:.2f}, "
            f"hold={wp.hold_steps}"
        )

    left_pos = [f"{x:.3f}" for x in initial_left_world[:3].tolist()]
    right_pos = [f"{x:.3f}" for x in initial_right_world[:3].tolist()]
    print(f"[WaypointRunner] Initial EE: left=[{','.join(left_pos)}], right=[{','.join(right_pos)}]")

    current_index = 0
    active_command = prepared_waypoints[current_index]
    ee_pose_left_w = left_arm.data.body_pose_w[:, left_entity.body_ids[0]]
    ee_pose_right_w = right_arm.data.body_pose_w[:, right_entity.body_ids[0]]
    ee_pos_left_b_init, ee_quat_left_b_init = subtract_frame_transforms(
        left_root_pose[:, 0:3], left_root_pose[:, 3:7], ee_pose_left_w[:, 0:3], ee_pose_left_w[:, 3:7]
    )
    ee_pos_right_b_init, ee_quat_right_b_init = subtract_frame_transforms(
        right_root_pose[:, 0:3], right_root_pose[:, 3:7], ee_pose_right_w[:, 0:3], ee_pose_right_w[:, 3:7]
    )

    if controls_orientation:
        desired_left_base = _convert_world_to_base(commanded_left_world.view(-1), left_root_pose).view(env.num_envs, -1)
        desired_right_base = _convert_world_to_base(commanded_right_world.view(-1), right_root_pose).view(env.num_envs, -1)
        left_ik.set_command(desired_left_base)
        right_ik.set_command(desired_right_base)
    else:
        # Position mode: only pass position, keep current orientation
        desired_left_pos_base = _convert_world_to_base(commanded_left_world.view(-1), left_root_pose).view(env.num_envs, -1)[:, :3]
        desired_right_pos_base = _convert_world_to_base(commanded_right_world.view(-1), right_root_pose).view(env.num_envs, -1)[:, :3]
        left_ik.set_command(desired_left_pos_base, ee_quat=ee_quat_left_b_init)
        right_ik.set_command(desired_right_pos_base, ee_quat=ee_quat_right_b_init)
    required_hold = max(args_cli.hold_steps, active_command.hold_steps)
    hold_counter = 0

    left_jacobian_index = left_entity.body_ids[0] - 1 if left_arm.is_fixed_base else left_entity.body_ids[0]
    right_jacobian_index = right_entity.body_ids[0] - 1 if right_arm.is_fixed_base else right_entity.body_ids[0]

    def _resolve_joint_indices(raw_ids, asset):
        if isinstance(raw_ids, slice):
            start = raw_ids.start or 0
            stop = raw_ids.stop or asset.num_joints
            step = raw_ids.step or 1
            return list(range(start, stop, step))
        return [int(idx) for idx in raw_ids]

    left_jacobian_joint_ids = _resolve_joint_indices(left_entity.joint_ids, left_arm)
    right_jacobian_joint_ids = _resolve_joint_indices(right_entity.joint_ids, right_arm)
    if not left_arm.is_fixed_base:
        left_jacobian_joint_ids = [idx + 6 for idx in left_jacobian_joint_ids]
    if not right_arm.is_fixed_base:
        right_jacobian_joint_ids = [idx + 6 for idx in right_jacobian_joint_ids]

    last_pos_log_time = time.time() if args_cli.logging_pos else None
    log_ik_targets = False  # Flag for logging IK computed targets

    print(f"[WaypointRunner] Loaded {len(prepared_waypoints)} waypoints. Starting execution...")

    total_steps = 0
    completed_all = False
    try:
        while simulation_app.is_running():
            if total_steps >= args_cli.max_steps:
                print(f"[WaypointRunner] Reached max step limit ({args_cli.max_steps}). Stopping.")
                break

            jacobian_left_w = left_arm.root_physx_view.get_jacobians()[:, left_jacobian_index, :, left_jacobian_joint_ids]
            jacobian_right_w = right_arm.root_physx_view.get_jacobians()[:, right_jacobian_index, :, right_jacobian_joint_ids]

            ee_pose_left_w = left_arm.data.body_pose_w[:, left_entity.body_ids[0]]
            ee_pose_right_w = right_arm.data.body_pose_w[:, right_entity.body_ids[0]]

            root_pose_left = left_arm.data.root_pose_w
            root_pose_right = right_arm.data.root_pose_w

            ee_pos_left_b, ee_quat_left_b = subtract_frame_transforms(
                root_pose_left[:, 0:3], root_pose_left[:, 3:7], ee_pose_left_w[:, 0:3], ee_pose_left_w[:, 3:7]
            )
            ee_pos_right_b, ee_quat_right_b = subtract_frame_transforms(
                root_pose_right[:, 0:3], root_pose_right[:, 3:7], ee_pose_right_w[:, 0:3], ee_pose_right_w[:, 3:7]
            )

            rot_left = matrix_from_quat(quat_inv(root_pose_left[:, 3:7]))
            rot_right = matrix_from_quat(quat_inv(root_pose_right[:, 3:7]))
            jacobian_left = jacobian_left_w.clone()
            jacobian_left[:, :3, :] = torch.bmm(rot_left, jacobian_left_w[:, :3, :])
            jacobian_left[:, 3:, :] = torch.bmm(rot_left, jacobian_left_w[:, 3:, :])
            jacobian_right = jacobian_right_w.clone()
            jacobian_right[:, :3, :] = torch.bmm(rot_right, jacobian_right_w[:, :3, :])
            jacobian_right[:, 3:, :] = torch.bmm(rot_right, jacobian_right_w[:, 3:, :])

            if controls_orientation:
                commanded_left_world = torch.lerp(commanded_left_world, active_command.left_world_pose, pose_gain)
                commanded_right_world = torch.lerp(commanded_right_world, active_command.right_world_pose, pose_gain)
                commanded_left_world[:, 3:] = _normalize_quat(commanded_left_world[:, 3:])
                commanded_right_world[:, 3:] = _normalize_quat(commanded_right_world[:, 3:])
                desired_left_base = _convert_world_to_base(commanded_left_world.view(-1), root_pose_left).view(env.num_envs, -1)
                desired_right_base = _convert_world_to_base(commanded_right_world.view(-1), root_pose_right).view(env.num_envs, -1)
                left_ik.set_command(desired_left_base)
                right_ik.set_command(desired_right_base)
            else:
                # Position mode: interpolate position only, keep current orientation
                commanded_left_pos = torch.lerp(commanded_left_world[:, :3], active_command.left_world_pose[:, :3], pose_gain)
                commanded_right_pos = torch.lerp(commanded_right_world[:, :3], active_command.right_world_pose[:, :3], pose_gain)
                commanded_left_world[:, :3] = commanded_left_pos
                commanded_right_world[:, :3] = commanded_right_pos
                desired_left_pos_base = _convert_world_to_base(commanded_left_world.view(-1), root_pose_left).view(env.num_envs, -1)[:, :3]
                desired_right_pos_base = _convert_world_to_base(commanded_right_world.view(-1), root_pose_right).view(env.num_envs, -1)[:, :3]
                left_ik.set_command(desired_left_pos_base, ee_quat=ee_quat_left_b)
                right_ik.set_command(desired_right_pos_base, ee_quat=ee_quat_right_b)

            left_pos_b_tensor = ee_pos_left_b[0, :3].detach().cpu()
            right_pos_b_tensor = ee_pos_right_b[0, :3].detach().cpu()
            if controls_orientation:
                target_left_b_tensor = desired_left_base[0, :3].detach().cpu()
                target_right_b_tensor = desired_right_base[0, :3].detach().cpu()
            else:
                target_left_b_tensor = desired_left_pos_base[0, :3].detach().cpu()
                target_right_b_tensor = desired_right_pos_base[0, :3].detach().cpu()

            if last_pos_log_time is not None:
                current_time = time.time()
                if current_time - last_pos_log_time >= 2.0:
                    left_pos = [f"{x:.3f}" for x in ee_pose_left_w[0, :3].detach().cpu().tolist()]
                    right_pos = [f"{x:.3f}" for x in ee_pose_right_w[0, :3].detach().cpu().tolist()]
                    target_left = [f"{x:.3f}" for x in active_command.left_world_pose[0, :3].detach().cpu().tolist()]
                    target_right = [f"{x:.3f}" for x in active_command.right_world_pose[0, :3].detach().cpu().tolist()]
                    left_err = (active_command.left_world_pose[0, :3] - ee_pose_left_w[0, :3]).norm().item()
                    right_err = (active_command.right_world_pose[0, :3] - ee_pose_right_w[0, :3]).norm().item()

                    # Joint angles (in radians, convert to degrees for readability)
                    left_joints_current = left_arm.data.joint_pos[:, left_entity.joint_ids][0].detach().cpu().tolist()
                    left_joints_current_deg = [f"{math.degrees(j):.1f}" for j in left_joints_current[:5]]  # First 5 are arm joints

                    print(
                        f"[WaypointRunner] WP{current_index+1}: "
                        f"L=[{','.join(left_pos)}]→[{','.join(target_left)}] pos_err={left_err:.3f} | "
                        f"R=[{','.join(right_pos)}]→[{','.join(target_right)}] pos_err={right_err:.3f}"
                    )

                    if controls_orientation:
                        # Orientation info
                        left_quat = [f"{x:.3f}" for x in ee_pose_left_w[0, 3:].detach().cpu().tolist()]
                        target_left_quat = [f"{x:.3f}" for x in active_command.left_world_pose[0, 3:].detach().cpu().tolist()]
                        # Calculate quaternion error
                        quat_dot_left = torch.abs(torch.sum(active_command.left_world_pose[:, 3:] * ee_pose_left_w[:, 3:], dim=-1)).item()
                        quat_error_left = 1.0 - quat_dot_left
                        print(
                            f"  L_quat: [{','.join(left_quat)}] → [{','.join(target_left_quat)}] quat_err={quat_error_left:.4f}"
                        )

                    print(
                        f"  L_joints_current (deg): [{','.join(left_joints_current_deg)}] (pan,lift,elbow,wrist_flex,wrist_roll)"
                    )
                    last_pos_log_time = current_time
                    log_ik_targets = True  # Flag to log IK targets after computation
                else:
                    log_ik_targets = False
            else:
                log_ik_targets = False

            joint_pos_left = left_arm.data.joint_pos[:, left_entity.joint_ids]
            joint_pos_right = right_arm.data.joint_pos[:, right_entity.joint_ids]

            joint_targets_left = left_ik.compute(ee_pos_left_b, ee_quat_left_b, jacobian_left, joint_pos_left)
            joint_targets_right = right_ik.compute(ee_pos_right_b, ee_quat_right_b, jacobian_right, joint_pos_right)

            # Log IK computed targets if flag is set
            if log_ik_targets:
                left_targets_deg = [f"{math.degrees(j):.1f}" for j in joint_targets_left[0, :5].detach().cpu().tolist()]
                left_delta_deg = [f"{math.degrees(joint_targets_left[0, i].item() - left_joints_current[i]):.1f}"
                                  for i in range(5)]
                print(
                    f"  L_joints_target  (deg): [{','.join(left_targets_deg)}] (pan,lift,elbow,wrist_flex,wrist_roll)"
                )
                print(
                    f"  L_joints_delta   (deg): [{','.join(left_delta_deg)}]"
                )

            if interp_gain < 1.0:
                joint_targets_left = joint_pos_left + interp_gain * (joint_targets_left - joint_pos_left)
                joint_targets_right = joint_pos_right + interp_gain * (joint_targets_right - joint_pos_right)

            if args_cli.debug_ik and total_steps < 20:
                joint_delta_left = (joint_targets_left - joint_pos_left)[0].detach().cpu().tolist()
                joint_delta_right = (joint_targets_right - joint_pos_right)[0].detach().cpu().tolist()
                print(
                    f"[WaypointRunner][Debug] step={total_steps} left_base_pos={left_pos_b_tensor.tolist()} "
                    f"target_left_base={target_left_b_tensor.tolist()} joint_delta_left={joint_delta_left}"
                )
                print(
                    f"[WaypointRunner][Debug] step={total_steps} right_base_pos={right_pos_b_tensor.tolist()} "
                    f"target_right_base={target_right_b_tensor.tolist()} joint_delta_right={joint_delta_right}"
                )

            action_tensor = _compose_bi_arm_action(
                joint_targets_left,
                active_command.left_gripper,
                joint_targets_right,
                active_command.right_gripper,
                arm_limits,
                gripper_limits,
            )
            _obs, *_ = env.step(action_tensor)

            # Check for convergence
            pos_err_left = torch.linalg.norm(active_command.left_world_pose[:, :3] - ee_pose_left_w[:, :3], dim=-1)
            pos_err_right = torch.linalg.norm(active_command.right_world_pose[:, :3] - ee_pose_right_w[:, :3], dim=-1)
            within_pos = torch.all(pos_err_left <= args_cli.position_tol) and torch.all(
                pos_err_right <= args_cli.position_tol
            )
            if controls_orientation:
                quat_err_left = 1.0 - torch.abs(torch.sum(active_command.left_world_pose[:, 3:] * ee_pose_left_w[:, 3:], dim=-1))
                quat_err_right = 1.0 - torch.abs(torch.sum(active_command.right_world_pose[:, 3:] * ee_pose_right_w[:, 3:], dim=-1))
                within_rot = torch.all(quat_err_left <= args_cli.orientation_tol) and torch.all(
                    quat_err_right <= args_cli.orientation_tol
                )
            else:
                quat_err_left = torch.zeros_like(pos_err_left)
                quat_err_right = torch.zeros_like(pos_err_right)
                within_rot = True

            if within_pos and within_rot:
                hold_counter += 1
                if hold_counter % 10 == 0:  # Log every 10 steps
                    print(
                        f"[WaypointRunner] WP{current_index+1} holding: {hold_counter}/{required_hold} "
                        f"pos_err=[{pos_err_left.item():.3f}, {pos_err_right.item():.3f}]"
                    )
            else:
                if hold_counter > 0:
                    print(
                        f"[WaypointRunner] WP{current_index+1} hold reset: "
                        f"pos_err=[{pos_err_left.item():.3f}, {pos_err_right.item():.3f}] "
                        f"(was at {hold_counter}/{required_hold})"
                    )
                hold_counter = 0

            if hold_counter >= required_hold:
                current_index += 1
                if current_index >= len(prepared_waypoints):
                    print("[WaypointRunner] Completed all waypoints.")
                    completed_all = True
                    break
                active_command = prepared_waypoints[current_index]
                # Use current EE world pose as starting point for next waypoint
                commanded_left_world = ee_pose_left_w.clone()
                commanded_right_world = ee_pose_right_w.clone()
                commanded_left_world[:, 3:] = _normalize_quat(commanded_left_world[:, 3:])
                commanded_right_world[:, 3:] = _normalize_quat(commanded_right_world[:, 3:])
                left_ik.reset()
                right_ik.reset()

                # Set initial command for new waypoint
                if controls_orientation:
                    # Pose mode: use current EE pose in base frame
                    desired_left_base = _convert_world_to_base(commanded_left_world.view(-1), root_pose_left).view(env.num_envs, -1)
                    desired_right_base = _convert_world_to_base(commanded_right_world.view(-1), root_pose_right).view(env.num_envs, -1)
                    left_ik.set_command(desired_left_base)
                    right_ik.set_command(desired_right_base)
                else:
                    # Position mode: only position, keep current orientation
                    ee_pos_left_b_new, ee_quat_left_b_new = subtract_frame_transforms(
                        root_pose_left[:, 0:3], root_pose_left[:, 3:7], ee_pose_left_w[:, 0:3], ee_pose_left_w[:, 3:7]
                    )
                    ee_pos_right_b_new, ee_quat_right_b_new = subtract_frame_transforms(
                        root_pose_right[:, 0:3], root_pose_right[:, 3:7], ee_pose_right_w[:, 0:3], ee_pose_right_w[:, 3:7]
                    )
                    desired_left_pos_base = _convert_world_to_base(commanded_left_world.view(-1), root_pose_left).view(env.num_envs, -1)[:, :3]
                    desired_right_pos_base = _convert_world_to_base(commanded_right_world.view(-1), root_pose_right).view(env.num_envs, -1)[:, :3]
                    left_ik.set_command(desired_left_pos_base, ee_quat=ee_quat_left_b_new)
                    right_ik.set_command(desired_right_pos_base, ee_quat=ee_quat_right_b_new)
                required_hold = max(args_cli.hold_steps, active_command.hold_steps)
                hold_counter = 0
                target_pos = [f"{x:.3f}" for x in active_command.left_world_pose[0, :3].detach().cpu().tolist()]
                print(f"[WaypointRunner] → WP{current_index + 1}/{len(prepared_waypoints)}: target=[{','.join(target_pos)}], hold={required_hold}")

            if rate_limiter:
                rate_limiter.sleep(env)
            total_steps += 1

    except Exception as e:
        print(f"[WaypointRunner] Exception occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if args_cli.stay_alive:
            status = "completed all waypoints" if completed_all else "stopped before completion"
            print(f"[WaypointRunner] Stay-alive requested ({status}). Leave this window open to keep sim running.")
            try:
                while simulation_app.is_running():
                    simulation_app.update()
            except KeyboardInterrupt:
                print("[WaypointRunner] Stay-alive loop interrupted by user.")
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()

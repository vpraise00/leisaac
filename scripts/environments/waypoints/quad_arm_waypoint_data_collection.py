# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect demonstration data using quad-arm waypoint execution."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Quad-arm waypoint-based data collection.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (must be 1).")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")

# Waypoint parameters
parser.add_argument("--controller_type", type=str, default="dik", choices=["dik"], help="Controller type: 'dik' (DifferentialIK).")
parser.add_argument("--waypoint_file", type=str, default=None, help="Path to JSON file with waypoints.")
parser.add_argument("--auto_table_legs", action="store_true", default=False, help="Auto-generate waypoints from mini table leg world poses.")
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument("--hold_steps", type=int, default=None, help="Override hold_steps for all waypoints.")
parser.add_argument("--episode_end_delay", type=float, default=2.0, help="Delay in seconds after last waypoint before ending episode.")
parser.add_argument("--position_tol", type=float, default=0.05, help="Position convergence tolerance.")
parser.add_argument("--orientation_tol", type=float, default=0.02, help="Orientation convergence tolerance.")
parser.add_argument("--pose_interp_gain", type=float, default=0.15, help="Pose interpolation gain.")
parser.add_argument("--interp_gain", type=float, default=0.15, help="Joint interpolation gain (only for dik).")
parser.add_argument("--command_type", type=str, default="position", choices=["position", "pose"], help="IK command type (only for dik).")
parser.add_argument("--force_wrist_down", action="store_true", default=False, help="Force wrist_flex joint to point downward.")
parser.add_argument("--wrist_flex_angle", type=float, default=1.57, help="Target angle for wrist_flex joint in radians (default: 1.57 = 90 deg).")
parser.add_argument("--wrist_flex_min", type=float, default=None, help="Minimum allowed wrist_flex angle in radians (boundary constraint).")
parser.add_argument("--wrist_flex_max", type=float, default=None, help="Maximum allowed wrist_flex angle in radians (boundary constraint).")

# Recording parameters
parser.add_argument("--record", action="store_true", default=False, help="Enable data recording.")
parser.add_argument("--dataset_file", type=str, default="./datasets/quad_waypoint_dataset.hdf5", help="HDF5 output file.")
parser.add_argument("--num_demos", type=int, default=10, help="Number of demonstrations to record.")
parser.add_argument("--episode_timeout", type=float, default=120.0, help="Maximum episode duration in seconds (default: 120s). Episode will restart if timeout is exceeded.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs != 1:
    raise ValueError("Waypoint data collection only supports num_envs=1")

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

try:
    REPO_ROOT = Path(__file__).resolve().parents[3]
except IndexError:
    REPO_ROOT = Path(__file__).resolve().parent
PACKAGE_DIR = REPO_ROOT / "source" / "leisaac"
if PACKAGE_DIR.is_dir():
    package_path_str = str(PACKAGE_DIR)
    if package_path_str not in sys.path:
        sys.path.insert(0, package_path_str)

import os
import time
import torch
import gymnasium as gym
import gc

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.managers import TerminationTermCfg

from leisaac.enhance.managers import StreamingRecorderManager
from quad_arm_waypoint_controller_dik import QuadArmWaypointController, load_waypoints_from_json, WaypointCommand
from leisaac.assets.scenes.collaborate_demo import MINI_TABLE_USD_PATH
from leisaac.utils import general_assets


class RateLimiter:
    """Rate limiter for control loop."""
    def __init__(self, hz):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz

    def sleep(self, env):
        next_wakeup_time = self.last_time + self.sleep_duration
        # Sleep directly without excessive rendering to save GPU memory
        remaining = next_wakeup_time - time.time()
        if remaining > 0:
            time.sleep(remaining)
        self.last_time += self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration

    def reset(self):
        self.last_time = time.time()


def main():
    """Main waypoint execution loop (with optional recording)."""

    # Parse environment config
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device("quad-so101leader")
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())

    # Disable time-based terminations
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None

    # Setup recording if enabled
    if args_cli.record:
        output_dir = os.path.dirname(args_cli.dataset_file)
        output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        if not hasattr(env_cfg.terminations, "success"):
            setattr(env_cfg.terminations, "success", None)
        env_cfg.terminations.success = TerminationTermCfg(
            func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        )
    else:
        env_cfg.recorders = None

    # Create environment
    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Setup recorder if recording enabled
    if args_cli.record:
        del env.recorder_manager
        env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
        env.recorder_manager.flush_steps = 50  # Flush frequently to prevent GPU memory buildup
        env.recorder_manager.compression = None  # Disable compression to save GPU/CPU memory
        print("[WaypointRunner] Environment created (recording enabled).")
        print(f"[WaypointRunner] Target: {args_cli.num_demos} demonstrations")
    else:
        print("[WaypointRunner] Environment created (playback only, no recording).")

    def pose_from_xyz(xyz):
        return torch.tensor([[xyz[0], xyz[1], xyz[2], 1.0, 0.0, 0.0, 0.0]], device=env.device, dtype=torch.float32)

    def assign_legs(legs_world: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Map found leg prims to north/east/west/south."""
        mapped: dict[str, torch.Tensor] = {}
        # 1) Try name-based mapping
        for name, pos in legs_world.items():
            lname = name.lower()
            if "north" in lname:
                mapped["north"] = pos
            elif "east" in lname:
                mapped["east"] = pos
            elif "west" in lname:
                mapped["west"] = pos
            elif "south" in lname:
                mapped["south"] = pos
        # 2) If not all four found, fall back to geometric mapping on xy
        if len(mapped) < 4:
            if len(legs_world) < 4:
                raise RuntimeError(f"Need 4 legs, found {len(legs_world)}: {list(legs_world.keys())}")
            # use first four entries
            items = list(legs_world.items())[:4]
            positions = torch.stack([p for _, p in items])
            center = positions.mean(dim=0)
            # compute angle from center
            angles = torch.atan2(positions[:, 1] - center[1], positions[:, 0] - center[0])  # rad
            # sort by angle: north ~ pi/2, south ~ -pi/2, east ~ 0, west ~ pi
            order = torch.argsort(angles)  # ascending
            sorted_items = [items[i] for i in order.tolist()]
            # pick by extreme y/x
            north = max(items, key=lambda kv: kv[1][1])
            south = min(items, key=lambda kv: kv[1][1])
            east = max(items, key=lambda kv: kv[1][0])
            west = min(items, key=lambda kv: kv[1][0])
            mapped = {"north": north[1], "east": east[1], "west": west[1], "south": south[1]}
        return mapped

    def generate_leg_waypoints(legs_world):
        # legs_world contains XYZ positions (no orientation)
        legs_world = assign_legs(legs_world)
        safe_offset = 0.15
        grip_offset = 0.02
        lift_delta = 0.10

        leg_offsets_outer = {
            "north": torch.tensor([0.0, 0.05, safe_offset], device=env.device),
            "south": torch.tensor([0.0, -0.05, safe_offset], device=env.device),
            "east": torch.tensor([0.05, 0.0, safe_offset], device=env.device),
            "west": torch.tensor([-0.05, 0.0, safe_offset], device=env.device),
        }

        legs_outer_xyz = {k: v + leg_offsets_outer[k] for k, v in legs_world.items()}
        legs_safe_xyz = {k: v + torch.tensor([0.0, 0.0, safe_offset], device=env.device) for k, v in legs_world.items()}
        legs_grip_xyz = {k: v + torch.tensor([0.0, 0.0, grip_offset], device=env.device) for k, v in legs_world.items()}
        legs_lift_xyz = {k: legs_grip_xyz[k] + torch.tensor([0.0, 0.0, lift_delta], device=env.device) for k in legs_world.keys()}

        legs_outer = {k: pose_from_xyz(v) for k, v in legs_outer_xyz.items()}
        legs_safe = {k: pose_from_xyz(v) for k, v in legs_safe_xyz.items()}
        legs_grip = {k: pose_from_xyz(v) for k, v in legs_grip_xyz.items()}
        legs_lift = {k: pose_from_xyz(v) for k, v in legs_lift_xyz.items()}
        return [
            # Extra align step outside legs to let wrist_roll settle
            WaypointCommand(legs_outer["north"], legs_outer["east"], legs_outer["west"], legs_outer["south"], 0.7, 0.7, 0.7, 0.7, 120),
            WaypointCommand(legs_safe["north"], legs_safe["east"], legs_safe["west"], legs_safe["south"], 0.7, 0.7, 0.7, 0.7, 80),
            WaypointCommand(legs_grip["north"], legs_grip["east"], legs_grip["west"], legs_grip["south"], 0.7, 0.7, 0.7, 0.7, 40),
            WaypointCommand(legs_grip["north"], legs_grip["east"], legs_grip["west"], legs_grip["south"], 0.0, 0.0, 0.0, 0.0, 40),
            WaypointCommand(legs_lift["north"], legs_lift["east"], legs_lift["west"], legs_lift["south"], 0.0, 0.0, 0.0, 0.0, 80),
            WaypointCommand(legs_grip["north"], legs_grip["east"], legs_grip["west"], legs_grip["south"], 0.0, 0.0, 0.0, 0.0, 40),
            WaypointCommand(legs_grip["north"], legs_grip["east"], legs_grip["west"], legs_grip["south"], 0.7, 0.7, 0.7, 0.7, 30),
            WaypointCommand(legs_safe["north"], legs_safe["east"], legs_safe["west"], legs_safe["south"], 0.7, 0.7, 0.7, 0.7, 40),
        ]

    if args_cli.auto_table_legs:
        table_usd = MINI_TABLE_USD_PATH
        stage = general_assets.get_stage(table_usd)
        legs_world = {}
        candidate_paths = [
            "/MiniTable/mini_table_instance/tableleg_north",
            "/MiniTable/mini_table_instance/tableleg_east",
            "/MiniTable/mini_table_instance/tableleg_west",
            "/MiniTable/mini_table_instance/tableleg_south",
            # fallback if USD doesn’t have the instance scope
            "/MiniTable/tableleg_north",
            "/MiniTable/tableleg_east",
            "/MiniTable/tableleg_west",
            "/MiniTable/tableleg_south",
        ]
        for path in candidate_paths:
            prim = stage.GetPrimAtPath(path)
            if prim:
                name = path.split("/")[-1].replace("tableleg_", "")
                pos, _ = general_assets.get_prim_pos_rot(prim)
                legs_world[name] = torch.tensor(pos, device=env.device, dtype=torch.float32)

        if len(legs_world) < 4:
            # Try generic search for any prim containing "tableleg"
            all_prims = general_assets.get_all_prims(stage)
            for prim in all_prims:
                pname = prim.GetPath().pathString
                if "tableleg" in pname.lower():
                    pos, _ = general_assets.get_prim_pos_rot(prim)
                    legs_world[pname.split("/")[-1]] = torch.tensor(pos, device=env.device, dtype=torch.float32)

        table_offset = torch.zeros(3, device=env.device)
        if hasattr(env_cfg.scene, "mini_table") and getattr(env_cfg.scene.mini_table, "init_state", None):
            if getattr(env_cfg.scene.mini_table.init_state, "pos", None):
                table_offset = torch.tensor(env_cfg.scene.mini_table.init_state.pos, device=env.device, dtype=torch.float32)
        for name in legs_world:
            legs_world[name] = legs_world[name] + table_offset

        if len(legs_world) < 4:
            print(f"[AutoTable] Could not find 4 tableleg prims in {table_usd}. Found: {list(legs_world.keys())}. Using default square around table center.")
            default = {
                "north": torch.tensor([0.15, 0.15, 0.0], device=env.device),
                "east": torch.tensor([0.15, -0.15, 0.0], device=env.device),
                "west": torch.tensor([-0.15, -0.15, 0.0], device=env.device),
                "south": torch.tensor([-0.15, 0.15, 0.0], device=env.device),
            }
            legs_world = {k: v + table_offset + torch.tensor([0.0, 0.0, 0.0], device=env.device) for k, v in default.items()}

        waypoints = generate_leg_waypoints(legs_world)
        waypoint_source = f"auto-generated from table legs ({table_usd})"
    else:
        if args_cli.waypoint_file is None:
            raise ValueError("waypoint_file is required unless --auto_table_legs is set")
        waypoints = load_waypoints_from_json(args_cli.waypoint_file, env.device)
        waypoint_source = args_cli.waypoint_file

    print(f"[DataCollection] Loaded {len(waypoints)} waypoints from {waypoint_source}")
    print("[DataCollection] Waypoint sequence:")
    for idx, wp in enumerate(waypoints, 1):
        print(f"  Waypoint {idx}:")
        print(f"    North:  pos=({wp.north_world_pose[0, 0]:.3f}, {wp.north_world_pose[0, 1]:.3f}, {wp.north_world_pose[0, 2]:.3f}), gripper={wp.north_gripper:.2f}")
        print(f"    East:   pos=({wp.east_world_pose[0, 0]:.3f}, {wp.east_world_pose[0, 1]:.3f}, {wp.east_world_pose[0, 2]:.3f}), gripper={wp.east_gripper:.2f}")
        print(f"    West:  pos=({wp.west_world_pose[0, 0]:.3f}, {wp.west_world_pose[0, 1]:.3f}, {wp.west_world_pose[0, 2]:.3f}), gripper={wp.west_gripper:.2f}")
        print(f"    South:   pos=({wp.south_world_pose[0, 0]:.3f}, {wp.south_world_pose[0, 1]:.3f}, {wp.south_world_pose[0, 2]:.3f}), gripper={wp.south_gripper:.2f}")
        print(f"    Hold steps: {wp.hold_steps}")
    print()

    # Create waypoint controller
    print("[DataCollection] Using DifferentialIK controller for quad-arm...")
    controller = QuadArmWaypointController(
        env=env,
        command_type=args_cli.command_type,
        position_tol=args_cli.position_tol,
        orientation_tol=args_cli.orientation_tol,
        pose_interp_gain=args_cli.pose_interp_gain,
        interp_gain=args_cli.interp_gain,
        force_wrist_down=args_cli.force_wrist_down,
        wrist_flex_angle=args_cli.wrist_flex_angle,
        wrist_flex_min=args_cli.wrist_flex_min,
        wrist_flex_max=args_cli.wrist_flex_max,
    )

    print("[DataCollection] Controller initialized.")
    print("[DataCollection] Episodes will run automatically.\n")

    rate_limiter = RateLimiter(args_cli.step_hz)

    # State
    current_demo_count = 0
    current_waypoint_index = 0
    episode_running = False
    episode_end_wait_time = None  # Time when last waypoint was reached
    episode_start_time = None  # Time when episode started

    # Reset environment
    env.reset()

    # Force initial GPU memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    rate_limiter.reset()

    # Start first episode
    if args_cli.record:
        print(f"[WaypointRunner] Starting Episode 1/{args_cli.num_demos}...")
    else:
        print(f"[WaypointRunner] Starting waypoint execution...")

    episode_running = True
    episode_start_time = time.time()
    current_waypoint_index = 0
    controller.reset()

    # Print initial waypoint info
    first_wp = waypoints[current_waypoint_index]
    print(f"\n{'='*70}")
    print(f"[WaypointRunner] Starting Waypoint 1/{len(waypoints)}")
    print(f"{'='*70}")
    print(f"  North Target: ({first_wp.north_world_pose[0, 0]:.3f}, {first_wp.north_world_pose[0, 1]:.3f}, {first_wp.north_world_pose[0, 2]:.3f}), gripper={first_wp.north_gripper:.2f}")
    print(f"  East  Target: ({first_wp.east_world_pose[0, 0]:.3f}, {first_wp.east_world_pose[0, 1]:.3f}, {first_wp.east_world_pose[0, 2]:.3f}), gripper={first_wp.east_gripper:.2f}")
    print(f"  West  Target: ({first_wp.west_world_pose[0, 0]:.3f}, {first_wp.west_world_pose[0, 1]:.3f}, {first_wp.west_world_pose[0, 2]:.3f}), gripper={first_wp.west_gripper:.2f}")
    print(f"  South Target: ({first_wp.south_world_pose[0, 0]:.3f}, {first_wp.south_world_pose[0, 1]:.3f}, {first_wp.south_world_pose[0, 2]:.3f}), gripper={first_wp.south_gripper:.2f}")
    print(f"  Hold steps: {args_cli.hold_steps if args_cli.hold_steps else first_wp.hold_steps}")
    print(f"{'='*70}\n")

    controller.set_waypoint(waypoints[current_waypoint_index], hold_steps_override=args_cli.hold_steps)

    # Main loop
    while simulation_app.is_running():
        with torch.inference_mode():
            # Check if waiting for episode end
            if episode_end_wait_time is not None:
                if time.time() - episode_end_wait_time >= args_cli.episode_end_delay:
                    # Delay complete, finish episode
                    if args_cli.record:
                        print(f"[WaypointRunner] Episode {current_demo_count + 1} complete!")

                        # Mark as successful
                        env.termination_manager.set_term_cfg(
                            "success",
                            TerminationTermCfg(func=lambda env: torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
                        )
                        env.termination_manager.compute()

                        # Reset environment
                        env.reset()

                        # Force GPU memory cleanup to prevent memory leak
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        gc.collect()

                        # Update demo count
                        if env.recorder_manager.exported_successful_episode_count > current_demo_count:
                            current_demo_count = env.recorder_manager.exported_successful_episode_count
                            print(f"[WaypointRunner] Progress: {current_demo_count}/{args_cli.num_demos} demonstrations.\n")

                        # Check if done
                        if current_demo_count >= args_cli.num_demos:
                            print(f"[WaypointRunner] All {args_cli.num_demos} demonstrations recorded!")
                            break

                        # Start next episode
                        print(f"[WaypointRunner] Starting Episode {current_demo_count + 1}/{args_cli.num_demos}...")
                        episode_running = True
                        episode_start_time = time.time()
                        current_waypoint_index = 0
                        episode_end_wait_time = None
                        controller.reset()

                        # Print initial waypoint info (next episode)
                        first_wp = waypoints[current_waypoint_index]
                        print(f"\n{'='*70}")
                        print(f"[WaypointRunner] Starting Waypoint 1/{len(waypoints)}")
                        print(f"{'='*70}")
                        print(f"  North Target: ({first_wp.north_world_pose[0, 0]:.3f}, {first_wp.north_world_pose[0, 1]:.3f}, {first_wp.north_world_pose[0, 2]:.3f}), gripper={first_wp.north_gripper:.2f}")
                        print(f"  East  Target: ({first_wp.east_world_pose[0, 0]:.3f}, {first_wp.east_world_pose[0, 1]:.3f}, {first_wp.east_world_pose[0, 2]:.3f}), gripper={first_wp.east_gripper:.2f}")
                        print(f"  West  Target: ({first_wp.west_world_pose[0, 0]:.3f}, {first_wp.west_world_pose[0, 1]:.3f}, {first_wp.west_world_pose[0, 2]:.3f}), gripper={first_wp.west_gripper:.2f}")
                        print(f"  South Target: ({first_wp.south_world_pose[0, 0]:.3f}, {first_wp.south_world_pose[0, 1]:.3f}, {first_wp.south_world_pose[0, 2]:.3f}), gripper={first_wp.south_gripper:.2f}")
                        print(f"  Hold steps: {args_cli.hold_steps if args_cli.hold_steps else first_wp.hold_steps}")
                        print(f"{'='*70}\n")

                        controller.set_waypoint(waypoints[current_waypoint_index], hold_steps_override=args_cli.hold_steps)

                        # Reset success termination
                        env.termination_manager.set_term_cfg(
                            "success",
                            TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
                        )
                        env.termination_manager.compute()
                    else:
                        # Playback mode - restart waypoints
                        print(f"[WaypointRunner] All waypoints complete. Restarting...")
                        env.reset()

                        # Force GPU memory cleanup
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        gc.collect()

                        episode_running = True
                        episode_start_time = time.time()
                        current_waypoint_index = 0
                        episode_end_wait_time = None
                        controller.reset()

                        # Print initial waypoint info (playback restart)
                        first_wp = waypoints[current_waypoint_index]
                        print(f"\n{'='*70}")
                        print(f"[WaypointRunner] Starting Waypoint 1/{len(waypoints)}")
                        print(f"{'='*70}")
                        print(f"  North Target: ({first_wp.north_world_pose[0, 0]:.3f}, {first_wp.north_world_pose[0, 1]:.3f}, {first_wp.north_world_pose[0, 2]:.3f}), gripper={first_wp.north_gripper:.2f}")
                        print(f"  East  Target: ({first_wp.east_world_pose[0, 0]:.3f}, {first_wp.east_world_pose[0, 1]:.3f}, {first_wp.east_world_pose[0, 2]:.3f}), gripper={first_wp.east_gripper:.2f}")
                        print(f"  West  Target: ({first_wp.west_world_pose[0, 0]:.3f}, {first_wp.west_world_pose[0, 1]:.3f}, {first_wp.west_world_pose[0, 2]:.3f}), gripper={first_wp.west_gripper:.2f}")
                        print(f"  South Target: ({first_wp.south_world_pose[0, 0]:.3f}, {first_wp.south_world_pose[0, 1]:.3f}, {first_wp.south_world_pose[0, 2]:.3f}), gripper={first_wp.south_gripper:.2f}")
                        print(f"  Hold steps: {args_cli.hold_steps if args_cli.hold_steps else first_wp.hold_steps}")
                        print(f"{'='*70}\n")

                        controller.set_waypoint(waypoints[current_waypoint_index], hold_steps_override=args_cli.hold_steps)
                else:
                    # Still waiting, just sleep
                    rate_limiter.sleep(env)
                    continue

            if episode_running:
                # Check episode timeout
                episode_duration = time.time() - episode_start_time
                if episode_duration > args_cli.episode_timeout:
                    print(f"[WaypointRunner] Episode timeout ({episode_duration:.1f}s > {args_cli.episode_timeout}s). Restarting episode...")

                    # Reset environment
                    env.reset()

                    # Force GPU memory cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()

                    # Restart episode
                    episode_running = True
                    episode_start_time = time.time()
                    current_waypoint_index = 0
                    episode_end_wait_time = None
                    controller.reset()

                    # Print initial waypoint info (timeout restart)
                    first_wp = waypoints[current_waypoint_index]
                    print(f"\n{'='*70}")
                    print(f"[WaypointRunner] Starting Waypoint 1/{len(waypoints)}")
                    print(f"{'='*70}")
                    print(f"  North Target: ({first_wp.north_world_pose[0, 0]:.3f}, {first_wp.north_world_pose[0, 1]:.3f}, {first_wp.north_world_pose[0, 2]:.3f}), gripper={first_wp.north_gripper:.2f}")
                    print(f"  East  Target: ({first_wp.east_world_pose[0, 0]:.3f}, {first_wp.east_world_pose[0, 1]:.3f}, {first_wp.east_world_pose[0, 2]:.3f}), gripper={first_wp.east_gripper:.2f}")
                    print(f"  West  Target: ({first_wp.west_world_pose[0, 0]:.3f}, {first_wp.west_world_pose[0, 1]:.3f}, {first_wp.west_world_pose[0, 2]:.3f}), gripper={first_wp.west_gripper:.2f}")
                    print(f"  South Target: ({first_wp.south_world_pose[0, 0]:.3f}, {first_wp.south_world_pose[0, 1]:.3f}, {first_wp.south_world_pose[0, 2]:.3f}), gripper={first_wp.south_gripper:.2f}")
                    print(f"  Hold steps: {args_cli.hold_steps if args_cli.hold_steps else first_wp.hold_steps}")
                    print(f"{'='*70}\n")

                    controller.set_waypoint(waypoints[current_waypoint_index], hold_steps_override=args_cli.hold_steps)
                    rate_limiter.sleep(env)
                    continue

                # Execute one control step
                action, converged = controller.step()
                env.step(action)

                if converged:
                    # Move to next waypoint
                    print(f"\n{'='*70}")
                    print(f"[WaypointRunner] Waypoint {current_waypoint_index + 1}/{len(waypoints)} COMPLETED!")
                    print(f"{'='*70}\n")

                    current_waypoint_index += 1

                    if current_waypoint_index >= len(waypoints):
                        # All waypoints reached - start delay timer
                        print(f"[WaypointRunner] All waypoints reached. Waiting {args_cli.episode_end_delay}s before ending episode...")
                        episode_end_wait_time = time.time()
                        episode_running = False
                    else:
                        # Next waypoint
                        next_wp = waypoints[current_waypoint_index]
                        print(f"{'='*70}")
                        print(f"[WaypointRunner] Moving to Waypoint {current_waypoint_index + 1}/{len(waypoints)}")
                        print(f"{'='*70}")
                        print(f"  North Target: ({next_wp.north_world_pose[0, 0]:.3f}, {next_wp.north_world_pose[0, 1]:.3f}, {next_wp.north_world_pose[0, 2]:.3f}), gripper={next_wp.north_gripper:.2f}")
                        print(f"  East  Target: ({next_wp.east_world_pose[0, 0]:.3f}, {next_wp.east_world_pose[0, 1]:.3f}, {next_wp.east_world_pose[0, 2]:.3f}), gripper={next_wp.east_gripper:.2f}")
                        print(f"  West  Target: ({next_wp.west_world_pose[0, 0]:.3f}, {next_wp.west_world_pose[0, 1]:.3f}, {next_wp.west_world_pose[0, 2]:.3f}), gripper={next_wp.west_gripper:.2f}")
                        print(f"  South Target: ({next_wp.south_world_pose[0, 0]:.3f}, {next_wp.south_world_pose[0, 1]:.3f}, {next_wp.south_world_pose[0, 2]:.3f}), gripper={next_wp.south_gripper:.2f}")
                        print(f"  Hold steps: {args_cli.hold_steps if args_cli.hold_steps else next_wp.hold_steps}")
                        print(f"{'='*70}\n")
                        controller.set_waypoint(waypoints[current_waypoint_index], hold_steps_override=args_cli.hold_steps)

            rate_limiter.sleep(env)

    # Cleanup
    print("[WaypointRunner] Closing...")
    env.close()
    simulation_app.close()
    print("[WaypointRunner] Done!")


if __name__ == "__main__":
    main()

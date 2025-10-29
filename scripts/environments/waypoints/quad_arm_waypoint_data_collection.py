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
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Quad-arm waypoint-based data collection.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (must be 1).")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")

# Waypoint parameters
parser.add_argument("--controller_type", type=str, default="dik", choices=["dik"], help="Controller type: 'dik' (DifferentialIK).")
parser.add_argument("--waypoint_file", type=str, required=True, help="Path to JSON file with waypoints.")
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument("--hold_steps", type=int, default=None, help="Override hold_steps for all waypoints.")
parser.add_argument("--episode_end_delay", type=float, default=2.0, help="Delay in seconds after last waypoint before ending episode.")
parser.add_argument("--position_tol", type=float, default=0.05, help="Position convergence tolerance.")
parser.add_argument("--orientation_tol", type=float, default=0.02, help="Orientation convergence tolerance.")
parser.add_argument("--pose_interp_gain", type=float, default=0.3, help="Pose interpolation gain.")
parser.add_argument("--interp_gain", type=float, default=0.3, help="Joint interpolation gain (only for dik).")
parser.add_argument("--command_type", type=str, default="position", choices=["position", "pose"], help="IK command type (only for dik).")
parser.add_argument("--force_wrist_down", action="store_true", help="Force wrist_flex joint to point downward.")
parser.add_argument("--wrist_flex_angle", type=float, default=1.57, help="Target angle for wrist_flex joint in radians (default: 1.57 = 90 deg).")
parser.add_argument("--max_joint_step", type=float, default=None, help="Maximum joint delta per control step (radians).")
parser.add_argument("--max_gripper_step", type=float, default=None, help="Maximum gripper delta per step.")

# Recording parameters
parser.add_argument("--record", action="store_true", default=False, help="Enable data recording.")
parser.add_argument("--dataset_file", type=str, default="./datasets/quad_waypoint_dataset.hdf5", help="HDF5 output file.")
parser.add_argument("--num_demos", type=int, default=10, help="Number of demonstrations to record.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs != 1:
    raise ValueError("Waypoint data collection only supports num_envs=1")

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import os
import time
import torch
import gymnasium as gym
import gc

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.managers import TerminationTermCfg

from leisaac.enhance.managers import StreamingRecorderManager
from quad_arm_waypoint_controller_dik import QuadArmWaypointController, load_waypoints_from_json


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

    # Load waypoints
    waypoints = load_waypoints_from_json(args_cli.waypoint_file, env.device)
    print(f"[DataCollection] Loaded {len(waypoints)} waypoints.")

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
        max_joint_delta=args_cli.max_joint_step,
        max_gripper_delta=args_cli.max_gripper_step,
    )

    print("[DataCollection] Controller initialized.")
    print("[DataCollection] Episodes will run automatically.\n")

    rate_limiter = RateLimiter(args_cli.step_hz)

    # State
    current_demo_count = 0
    current_waypoint_index = 0
    episode_running = False
    episode_end_wait_time = None  # Time when last waypoint was reached

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
    current_waypoint_index = 0
    controller.reset()
    controller.set_waypoint(waypoints[current_waypoint_index], hold_steps_override=args_cli.hold_steps)
    print(f"  → Waypoint 1/{len(waypoints)}")

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
                        current_waypoint_index = 0
                        episode_end_wait_time = None
                        controller.reset()
                        controller.set_waypoint(waypoints[current_waypoint_index], hold_steps_override=args_cli.hold_steps)
                        print(f"  → Waypoint 1/{len(waypoints)}")

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
                        current_waypoint_index = 0
                        episode_end_wait_time = None
                        controller.reset()
                        controller.set_waypoint(waypoints[current_waypoint_index], hold_steps_override=args_cli.hold_steps)
                        print(f"  → Waypoint 1/{len(waypoints)}")
                else:
                    # Still waiting, just sleep
                    rate_limiter.sleep(env)
                    continue

            if episode_running:
                # Execute one control step
                action, converged = controller.step()
                env.step(action)

                if converged:
                    # Move to next waypoint
                    current_waypoint_index += 1

                    if current_waypoint_index >= len(waypoints):
                        # All waypoints reached - check if desk is lifted before ending
                        desk = env.scene["desk"]
                        desk_height = desk.data.root_pos_w[0, 2].item()  # Get z-coordinate of desk
                        initial_desk_height = 0.05  # From env config

                        if desk_height > initial_desk_height + 0.01:  # Check if desk lifted by at least 1cm
                            print(f"[WaypointRunner] Desk lifted successfully (height: {desk_height:.3f}m). Waiting {args_cli.episode_end_delay}s before ending episode...")
                            episode_end_wait_time = time.time()
                            episode_running = False
                        else:
                            print(f"[WaypointRunner] Warning: Desk not lifted enough (height: {desk_height:.3f}m vs initial: {initial_desk_height:.3f}m). Holding at final waypoint...")
                            # Stay at final waypoint and keep trying to lift
                            current_waypoint_index = len(waypoints) - 1
                            controller.set_waypoint(waypoints[current_waypoint_index], hold_steps_override=args_cli.hold_steps)
                    else:
                        # Next waypoint
                        print(f"  → Waypoint {current_waypoint_index + 1}/{len(waypoints)}")
                        controller.set_waypoint(waypoints[current_waypoint_index], hold_steps_override=args_cli.hold_steps)

            rate_limiter.sleep(env)

    # Cleanup
    print("[WaypointRunner] Closing...")
    env.close()
    simulation_app.close()
    print("[WaypointRunner] Done!")


if __name__ == "__main__":
    main()

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect demonstration data using bi-arm waypoint execution."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Bi-arm waypoint-based data collection.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (must be 1).")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")

# Waypoint parameters
parser.add_argument("--controller_type", type=str, default="dik", choices=["dik", "osc"], help="Controller type: 'dik' (DifferentialIK) or 'osc' (OperationalSpaceController).")
parser.add_argument("--waypoint_file", type=str, required=True, help="Path to JSON file with waypoints.")
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument("--hold_steps", type=int, default=None, help="Override hold_steps for all waypoints.")
parser.add_argument("--episode_end_delay", type=float, default=2.0, help="Delay in seconds after last waypoint before ending episode.")
parser.add_argument("--position_tol", type=float, default=0.05, help="Position convergence tolerance.")
parser.add_argument("--orientation_tol", type=float, default=0.02, help="Orientation convergence tolerance.")
parser.add_argument("--pose_interp_gain", type=float, default=0.3, help="Pose interpolation gain.")
parser.add_argument("--interp_gain", type=float, default=0.3, help="Joint interpolation gain (only for dik).")
parser.add_argument("--command_type", type=str, default="position", choices=["position", "pose"], help="IK command type (only for dik).")
parser.add_argument("--motion_stiffness", type=float, default=150.0, help="Motion stiffness for OSC (only for osc).")
parser.add_argument("--motion_damping_ratio", type=float, default=1.0, help="Motion damping ratio for OSC (only for osc).")
parser.add_argument("--force_wrist_down", action="store_true", default=True, help="Force wrist_flex joint to point downward.")
parser.add_argument("--wrist_flex_angle", type=float, default=1.57, help="Target angle for wrist_flex joint in radians (default: 1.57 = 90 deg).")

# Recording parameters
parser.add_argument("--record", action="store_true", default=False, help="Enable data recording.")
parser.add_argument("--dataset_file", type=str, default="./datasets/waypoint_dataset.hdf5", help="HDF5 output file.")
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

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.managers import TerminationTermCfg

from leisaac.enhance.managers import StreamingRecorderManager
from waypoint_controller_dik import BiArmWaypointController, load_waypoints_from_json
from waypoint_controller_osc import BiArmWaypointControllerOSC, load_waypoints_from_json as load_waypoints_from_json_osc


class RateLimiter:
    """Rate limiter for control loop."""
    def __init__(self, hz):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
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
    env_cfg.use_teleop_device("bi-so101leader")
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
        env.recorder_manager.flush_steps = 100
        env.recorder_manager.compression = 'lzf'
        print("[WaypointRunner] Environment created (recording enabled).")
        print(f"[WaypointRunner] Target: {args_cli.num_demos} demonstrations")
    else:
        print("[WaypointRunner] Environment created (playback only, no recording).")

    # Load waypoints
    if args_cli.controller_type == "dik":
        waypoints = load_waypoints_from_json(args_cli.waypoint_file, env.device)
    else:  # osc
        waypoints = load_waypoints_from_json_osc(args_cli.waypoint_file, env.device)
    print(f"[DataCollection] Loaded {len(waypoints)} waypoints.")

    # Create waypoint controller based on type
    if args_cli.controller_type == "dik":
        print("[DataCollection] Using DifferentialIK controller...")
        controller = BiArmWaypointController(
            env=env,
            command_type=args_cli.command_type,
            position_tol=args_cli.position_tol,
            orientation_tol=args_cli.orientation_tol,
            pose_interp_gain=args_cli.pose_interp_gain,
            interp_gain=args_cli.interp_gain,
            force_wrist_down=args_cli.force_wrist_down,
            wrist_flex_angle=args_cli.wrist_flex_angle,
        )
    else:  # osc
        print("[DataCollection] Using OperationalSpaceController...")
        # Convert command_type for OSC
        osc_command_type = "pose_abs" if args_cli.command_type == "pose" else "pose_abs"
        controller = BiArmWaypointControllerOSC(
            env=env,
            command_type=osc_command_type,
            position_tol=args_cli.position_tol,
            orientation_tol=args_cli.orientation_tol,
            pose_interp_gain=args_cli.pose_interp_gain,
            motion_stiffness=args_cli.motion_stiffness,
            motion_damping_ratio=args_cli.motion_damping_ratio,
            force_wrist_down=args_cli.force_wrist_down,
            wrist_flex_angle=args_cli.wrist_flex_angle,
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
                        episode_running = True
                        current_waypoint_index = 0
                        episode_end_wait_time = None
                        controller.reset()
                        controller.set_waypoint(waypoints[current_waypoint_index], hold_steps_override=args_cli.hold_steps)
                        print(f"  → Waypoint 1/{len(waypoints)}")
                else:
                    # Still waiting, just step with last action
                    env.sim.render()
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
                        # All waypoints reached - start delay timer
                        print(f"[WaypointRunner] All waypoints reached. Waiting {args_cli.episode_end_delay}s before ending episode...")
                        episode_end_wait_time = time.time()
                        episode_running = False
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
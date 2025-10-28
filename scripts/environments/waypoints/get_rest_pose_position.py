"""Get end-effector position for LeRobot rest pose."""

import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="LeIsaac-SO101-CleanToyTable-BiArm-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(vars(args))
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

# Create env
env_cfg = parse_env_cfg(args.task, device="cuda", num_envs=1)
env_cfg.use_teleop_device("bi-so101leader")
env: ManagerBasedRLEnv = gym.make(args.task, cfg=env_cfg).unwrapped

# Reset to default pose
env.reset()
env.step(torch.zeros(1, 12, device="cuda"))

print("\n=== Default Pose (all joints = 0) ===")
print(f"Left arm joint pos: {env.scene['left_arm'].data.joint_pos[0, :6].cpu().numpy()}")
print(f"Right arm joint pos: {env.scene['right_arm'].data.joint_pos[0, :6].cpu().numpy()}")
left_ee_default = env.scene['left_arm'].data.body_pos_w[0, -1].cpu().numpy()
right_ee_default = env.scene['right_arm'].data.body_pos_w[0, -1].cpu().numpy()
print(f"Left EE position: {left_ee_default}")
print(f"Right EE position: {right_ee_default}")

# Set to LeRobot rest pose
rest_pose = torch.tensor([[
    0.0,        # shoulder_pan: 0 degree
    -1.50,      # shoulder_lift: -86 degree
    1.40,       # elbow_flex: 80 degree
    0.87,       # wrist_flex: 50 degree
    0.0,        # wrist_roll: 0 degree
    0.0         # gripper: 0 degree
]], device="cuda")

env.scene['left_arm'].write_joint_state_to_sim(rest_pose, rest_pose * 0)
env.scene['right_arm'].write_joint_state_to_sim(rest_pose, rest_pose * 0)

# Step simulation
for _ in range(10):
    env.sim.step()

print("\n=== LeRobot Rest Pose ===")
print(f"Left arm joint pos: {env.scene['left_arm'].data.joint_pos[0, :6].cpu().numpy()}")
print(f"Right arm joint pos: {env.scene['right_arm'].data.joint_pos[0, :6].cpu().numpy()}")
left_ee_rest = env.scene['left_arm'].data.body_pos_w[0, -1].cpu().numpy()
right_ee_rest = env.scene['right_arm'].data.body_pos_w[0, -1].cpu().numpy()
print(f"Left EE position: {left_ee_rest}")
print(f"Right EE position: {right_ee_rest}")

print("\n=== Suggested First Waypoint ===")
print(f'''{{
  "relative": false,
  "left": {{
    "position": [{left_ee_rest[0]:.2f}, {left_ee_rest[1]:.2f}, {left_ee_rest[2]:.2f}],
    "gripper": 0.0
  }},
  "right": {{
    "position": [{right_ee_rest[0]:.2f}, {right_ee_rest[1]:.2f}, {right_ee_rest[2]:.2f}],
    "gripper": 0.0
  }},
  "hold_steps": 45
}}''')

env.close()
simulation_app.close()

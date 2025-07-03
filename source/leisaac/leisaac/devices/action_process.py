import torch
from typing import Any

import isaaclab.envs.mdp as mdp

from leisaac.assets.robots.lerobot import SO101_FOLLOWER_USD_JOINT_LIMLITS


def init_action_cfg(action_cfg, device='keyboard'):
    if device in ['so101leader']:
        action_cfg.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=1.0,
        )
        action_cfg.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=1.0,
        )
    elif device in ['keyboard']:
        action_cfg.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=1.0,
        )
        action_cfg.gripper_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=0.7,
        )
    else:
        action_cfg.arm_action = None
        action_cfg.gripper_action = None
    return action_cfg


joint_names_to_motor_ids = {
    "shoulder_pan": 0,
    "shoulder_lift": 1,
    "elbow_flex": 2,
    "wrist_flex": 3,
    "wrist_roll": 4,
    "gripper": 5,
}

def preprocess_device_action(action: dict[str, Any], teleop_device) -> torch.Tensor:
    num_envs = teleop_device.env.num_envs
    if action.get('so101_leader') is not None:
        processed_action = torch.zeros(num_envs, 6, device=teleop_device.env.device)
        joint_limits = SO101_FOLLOWER_USD_JOINT_LIMLITS
        joint_state, motor_limits = action['joint_state'], action['motor_limits']
        for joint_name, motor_id in joint_names_to_motor_ids.items():
            motor_limit_range = motor_limits[joint_name]
            joint_limit_range = joint_limits[joint_name]
            processed_degree = (joint_state[joint_name] - motor_limit_range[0]) / (motor_limit_range[1] - motor_limit_range[0]) \
                * (joint_limit_range[1] - joint_limit_range[0]) + joint_limit_range[0]
            processed_radius = processed_degree / 180.0 * torch.pi # convert degree to radius
            processed_action[:, motor_id] = processed_radius
    elif action.get('keyboard') is not None:
        processed_action = torch.zeros(num_envs, 6, device=teleop_device.env.device)
        processed_action[:, :] = action['joint_state']
    else:
        raise NotImplementedError("Only teleoperation with so101_leader, keyboard is supported for now.")
    return processed_action

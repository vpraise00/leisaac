import torch
from typing import Any

import isaaclab.envs.mdp as mdp

from leisaac.assets.robots.lerobot import SO101_FOLLOWER_USD_JOINT_LIMLITS


def init_action_cfg(action_cfg, device):
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
    elif device in ['bi-so101leader']:
        action_cfg.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="left_arm",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=1.0,
        )
        action_cfg.left_gripper_action = mdp.JointPositionActionCfg(
            asset_name="left_arm",
            joint_names=["gripper"],
            scale=1.0,
        )
        action_cfg.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="right_arm",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=1.0,
        )
        action_cfg.right_gripper_action = mdp.JointPositionActionCfg(
            asset_name="right_arm",
            joint_names=["gripper"],
            scale=1.0,
        )
<<<<<<< Updated upstream
    elif device in ['mimic_so101leader']:
=======
    elif device in ['quad-so101leader']:
        action_cfg.nord_arm_action = mdp.JointPositionActionCfg(
            asset_name="nord_arm",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=1.0,
        )
        action_cfg.nord_gripper_action = mdp.JointPositionActionCfg(
            asset_name="nord_arm",
            joint_names=["gripper"],
            scale=1.0,
        )
        action_cfg.ost_arm_action = mdp.JointPositionActionCfg(
            asset_name="ost_arm",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=1.0,
        )
        action_cfg.ost_gripper_action = mdp.JointPositionActionCfg(
            asset_name="ost_arm",
            joint_names=["gripper"],
            scale=1.0,
        )
        action_cfg.west_arm_action = mdp.JointPositionActionCfg(
            asset_name="west_arm",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=1.0,
        )
        action_cfg.west_gripper_action = mdp.JointPositionActionCfg(
            asset_name="west_arm",
            joint_names=["gripper"],
            scale=1.0,
        )
        action_cfg.sud_arm_action = mdp.JointPositionActionCfg(
            asset_name="sud_arm",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=1.0,
        )
        action_cfg.sud_gripper_action = mdp.JointPositionActionCfg(
            asset_name="sud_arm",
            joint_names=["gripper"],
            scale=1.0,
        )
    elif device in ['mimic_ik_abs_so101_leader']:
>>>>>>> Stashed changes
        action_cfg.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            body_name="gripper",
            controller=mdp.DifferentialIKControllerCfg(command_type="pose", ik_method="dls", use_relative_mode=False),
        )
        action_cfg.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=1.0,
        )
    elif device in ['mimic_keyboard']:
        action_cfg.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            body_name="gripper",
            controller=mdp.DifferentialIKControllerCfg(command_type="pose", ik_method="dls", use_relative_mode=False),
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


def convert_action_from_so101_leader(joint_state: dict[str, float], motor_limits: dict[str, tuple[float, float]], teleop_device) -> torch.Tensor:
    processed_action = torch.zeros(teleop_device.env.num_envs, 6, device=teleop_device.env.device)
    joint_limits = SO101_FOLLOWER_USD_JOINT_LIMLITS
    for joint_name, motor_id in joint_names_to_motor_ids.items():
        motor_limit_range = motor_limits[joint_name]
        joint_limit_range = joint_limits[joint_name]
        processed_degree = (joint_state[joint_name] - motor_limit_range[0]) / (motor_limit_range[1] - motor_limit_range[0]) \
            * (joint_limit_range[1] - joint_limit_range[0]) + joint_limit_range[0]
        processed_radius = processed_degree / 180.0 * torch.pi  # convert degree to radius
        processed_action[:, motor_id] = processed_radius
    return processed_action


def preprocess_device_action(action: dict[str, Any], teleop_device) -> torch.Tensor:
    if action.get('so101_leader') is not None:
        processed_action = convert_action_from_so101_leader(action['joint_state'], action['motor_limits'], teleop_device)
    elif action.get('keyboard') is not None:
        processed_action = torch.zeros(teleop_device.env.num_envs, 6, device=teleop_device.env.device)
        processed_action[:, :] = action['joint_state']
    elif action.get('bi_so101_leader') is not None:
        processed_action = torch.zeros(teleop_device.env.num_envs, 12, device=teleop_device.env.device)
        processed_action[:, :6] = convert_action_from_so101_leader(action['joint_state']['left_arm'], action['motor_limits']['left_arm'], teleop_device)
        processed_action[:, 6:] = convert_action_from_so101_leader(action['joint_state']['right_arm'], action['motor_limits']['right_arm'], teleop_device)
    elif action.get('quad_so101_leader') is not None:
        processed_action = torch.zeros(teleop_device.env.num_envs, 24, device=teleop_device.env.device)
        processed_action[:, :6] = convert_action_from_so101_leader(action['joint_state']['nord_arm'], action['motor_limits']['nord_arm'], teleop_device)
        processed_action[:, 6:12] = convert_action_from_so101_leader(action['joint_state']['ost_arm'], action['motor_limits']['ost_arm'], teleop_device)
        processed_action[:, 12:18] = convert_action_from_so101_leader(action['joint_state']['west_arm'], action['motor_limits']['west_arm'], teleop_device)
        processed_action[:, 18:24] = convert_action_from_so101_leader(action['joint_state']['sud_arm'], action['motor_limits']['sud_arm'], teleop_device)
    else:
        raise NotImplementedError("Only teleoperation with so101_leader, bi_so101_leader, quad_so101_leader, keyboard is supported for now.")
    return processed_action

import torch
import math

from dataclasses import MISSING
from typing import Any

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg as RecordTerm
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from leisaac.assets.robots.lerobot import SO101_FOLLOWER_CFG
from leisaac.assets.scenes.desk_lift import DESK_LIFT_CFG, DESK_LIFT_USD_PATH, DESK_CFG, DESK_USD_PATH
from leisaac.devices.action_process import init_action_cfg, preprocess_device_action
from leisaac.utils.general_assets import parse_usd_and_create_subassets


def euler_deg_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple:
    """Convert euler angles in degrees to quaternion (w, x, y, z).

    Args:
        roll_deg: Rotation around X-axis in degrees
        pitch_deg: Rotation around Y-axis in degrees
        yaw_deg: Rotation around Z-axis in degrees

    Returns:
        Quaternion as (w, x, y, z) tuple
    """
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return (w, x, y, z)


@configclass
class LiftDeskQuadArmSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the lift desk task using four arms."""

    scene: AssetBaseCfg = DESK_LIFT_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    desk: RigidObjectCfg = DESK_CFG.replace(prim_path="{ENV_REGEX_NS}/Desk")

    nord_arm: ArticulationCfg = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/Nord_Robot")

    ost_arm: ArticulationCfg = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/Ost_Robot")

    west_arm: ArticulationCfg = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/West_Robot")

    sud_arm: ArticulationCfg = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/Sud_Robot")

    nord_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Nord_Robot/gripper/nord_wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.001, 0.1, -0.04), rot=(-0.404379, -0.912179, -0.0451242, 0.0486914), convention="ros"),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,  # For a 75° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=False
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )

    ost_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Ost_Robot/gripper/ost_wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.001, 0.1, -0.04), rot=(-0.404379, -0.912179, -0.0451242, 0.0486914), convention="ros"),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,  # For a 75° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=False
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )

    west_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/West_Robot/gripper/west_wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.001, 0.1, -0.04), rot=(-0.404379, -0.912179, -0.0451242, 0.0486914), convention="ros"),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,  # For a 75° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=False
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )

    sud_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Sud_Robot/gripper/sud_wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.001, 0.1, -0.04), rot=(-0.404379, -0.912179, -0.0451242, 0.0486914), convention="ros"),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,  # For a 75° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=False
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )

    top: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Scene/top_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, -1.0, 0.0, 0.0), convention="ros"),  # wxyz - looking down from above
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7,
            focus_distance=400.0,
            horizontal_aperture=28.11,  # For a 75° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=True
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )

    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Configuration for the actions."""
    nord_arm_action: mdp.ActionTermCfg = MISSING
    nord_gripper_action: mdp.ActionTermCfg = MISSING
    ost_arm_action: mdp.ActionTermCfg = MISSING
    ost_gripper_action: mdp.ActionTermCfg = MISSING
    west_arm_action: mdp.ActionTermCfg = MISSING
    west_gripper_action: mdp.ActionTermCfg = MISSING
    sud_arm_action: mdp.ActionTermCfg = MISSING
    sud_gripper_action: mdp.ActionTermCfg = MISSING


@configclass
class EventCfg:
    """Configuration for the events."""

    # reset to default scene
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # reset desk to initial position
    reset_desk = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("desk"),
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        nord_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("nord_arm")})
        nord_joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("nord_arm")})
        nord_joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("nord_arm")})
        nord_joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("nord_arm")})

        ost_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("ost_arm")})
        ost_joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("ost_arm")})
        ost_joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("ost_arm")})
        ost_joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("ost_arm")})

        west_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("west_arm")})
        west_joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("west_arm")})
        west_joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("west_arm")})
        west_joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("west_arm")})

        sud_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("sud_arm")})
        sud_joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("sud_arm")})
        sud_joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("sud_arm")})
        sud_joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("sud_arm")})

        actions = ObsTerm(func=mdp.last_action)
        nord = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("nord_wrist"), "data_type": "rgb", "normalize": False})
        ost = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("ost_wrist"), "data_type": "rgb", "normalize": False})
        west = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("west_wrist"), "data_type": "rgb", "normalize": False})
        sud = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("sud_wrist"), "data_type": "rgb", "normalize": False})
        top = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("top"), "data_type": "rgb", "normalize": False})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Configuration for the rewards"""


@configclass
class TerminationsCfg:
    """Configuration for the termination"""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class LiftDeskQuadArmEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lift desk environment."""

    scene: LiftDeskQuadArmSceneCfg = LiftDeskQuadArmSceneCfg(env_spacing=8.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    recorders: RecordTerm = RecordTerm()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.decimation = 1
        self.episode_length_s = 8.0
        self.viewer.eye = (-1.5, -2.0, 1.5)
        self.viewer.lookat = (-0.2, -0.3, 0.5)

        # PhysX collision settings to prevent tunneling
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_collision_stack_size = 2**28  # Increase for complex collisions
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**24
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**20
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**20
        self.sim.physx.solver_position_iteration_count = 16  # Further increase solver iterations to prevent tunneling
        self.sim.physx.solver_velocity_iteration_count = 8
        self.sim.physx.gpu_max_rigid_contact_count = 2**23  # Increase contact capacity
        self.sim.physx.gpu_max_rigid_patch_count = 2**21
        self.sim.render.enable_translucency = True

        # Position and orientation for 4 arms per user specification
        # Position: pos = (x, y, z) - scaled 1.5x from origin (0, 0)
        # Orientation: euler angles (roll, pitch, yaw) in degrees
        self.scene.nord_arm.init_state.pos = (0.45, -0.18975, 0.01)
        self.scene.nord_arm.init_state.rot = euler_deg_to_quat(0, 0, 0)  # 0° yaw

        self.scene.ost_arm.init_state.pos = (0.9, -0.6, 0.01)  # (0.6*1.5, -0.4*1.5)
        self.scene.ost_arm.init_state.rot = euler_deg_to_quat(0, 0, -90)  # -90° yaw

        self.scene.west_arm.init_state.pos = (0.0795, -0.6, 0.01)  # (0.053*1.5, -0.4*1.5)
        self.scene.west_arm.init_state.rot = euler_deg_to_quat(0, 0, 90)  # 90° yaw

        self.scene.sud_arm.init_state.pos = (0.45, -1.01025, 0.01)
        self.scene.sud_arm.init_state.rot = euler_deg_to_quat(0, 0, 180)  # 180° yaw

        # Position desk at the center of the 4 robots
        # Center calculation: average of all robot positions
        center_x = (0.45 + 0.9 + 0.0795 + 0.45) / 4  # ≈ 0.46988
        center_y = (-0.18975 + -0.6 + -0.6 + -1.01025) / 4  # ≈ -0.6
        desk_center_y = center_y + 0.02  # shift desk slightly north
        self.scene.desk.init_state.pos = (center_x, desk_center_y, 0.05)

        # Position top camera above desk center, looking down
        self.scene.top.offset.pos = (center_x, desk_center_y, 1.0)

        parse_usd_and_create_subassets(DESK_LIFT_USD_PATH, self)

    def use_teleop_device(self, teleop_device) -> None:
        self.actions = init_action_cfg(self.actions, device=teleop_device)
        if teleop_device == "keyboard":
            self.scene.robot.spawn.rigid_props.disable_gravity = True

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

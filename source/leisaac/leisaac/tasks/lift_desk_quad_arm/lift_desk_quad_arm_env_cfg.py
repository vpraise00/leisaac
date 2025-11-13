import torch
import math

from dataclasses import MISSING
from typing import Any

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg as RecordTerm
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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
from leisaac.assets.scenes.desk_lift import DESK_LIFT_CFG, DESK_LIFT_USD_PATH, DESK_CFG
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

    # desk: AssetBaseCfg = DESK_CFG.replace(prim_path="{ENV_REGEX_NS}/Desk")

    north_arm: ArticulationCfg = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/North_Robot")

    east_arm: ArticulationCfg = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/East_Robot")

    west_arm: ArticulationCfg = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/West_Robot")

    south_arm: ArticulationCfg = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/South_Robot")

    north_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/North_Robot/gripper/north_wrist_camera",
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

    east_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/East_Robot/gripper/east_wrist_camera",
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

    south_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/South_Robot/gripper/south_wrist_camera",
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

    # South top camera - positioned above south robot, looking straight down
    south_top: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/SouthTopCamera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.4, -0.85, 1.2), rot=(0.0, -1.0, 0.0, 0.0), convention="ros"),  # wxyz - Looking straight down
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,  # For wider FOV to capture desk and robots
            clipping_range=(0.01, 50.0),
            lock_camera=True
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )

    # North top camera - positioned above north robot, looking straight down
    north_top: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/NorthTopCamera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.4, -0.05, 1.2), rot=(0.0, -1.0, 0.0, 0.0), convention="ros"),  # wxyz - Looking straight down
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,  # For wider FOV to capture desk and robots
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
    north_arm_action: mdp.ActionTermCfg = MISSING
    north_gripper_action: mdp.ActionTermCfg = MISSING
    east_arm_action: mdp.ActionTermCfg = MISSING
    east_gripper_action: mdp.ActionTermCfg = MISSING
    west_arm_action: mdp.ActionTermCfg = MISSING
    west_gripper_action: mdp.ActionTermCfg = MISSING
    south_arm_action: mdp.ActionTermCfg = MISSING
    south_gripper_action: mdp.ActionTermCfg = MISSING


@configclass
class EventCfg:
    """Configuration for the events."""

    # reset to default scene
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # reset desk position
    reset_desk = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
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

        north_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("north_arm")})
        north_joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("north_arm")})
        north_joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("north_arm")})
        north_joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("north_arm")})

        east_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("east_arm")})
        east_joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("east_arm")})
        east_joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("east_arm")})
        east_joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("east_arm")})

        west_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("west_arm")})
        west_joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("west_arm")})
        west_joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("west_arm")})
        west_joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("west_arm")})

        south_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("south_arm")})
        south_joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("south_arm")})
        south_joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("south_arm")})
        south_joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("south_arm")})

        actions = ObsTerm(func=mdp.last_action)
        north = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("north_wrist"), "data_type": "rgb", "normalize": False})
        east = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("east_wrist"), "data_type": "rgb", "normalize": False})
        west = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("west_wrist"), "data_type": "rgb", "normalize": False})
        south = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("south_wrist"), "data_type": "rgb", "normalize": False})
        south_top = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("south_top"), "data_type": "rgb", "normalize": False})
        north_top = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("north_top"), "data_type": "rgb", "normalize": False})

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

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.friction_offset_threshold = 0.001  # Lower threshold for better friction
        self.sim.physx.enable_stabilization = True  # Improve contact stability
        self.sim.render.enable_translucency = True

        # Position and orientation for 4 arms in square formation
        # Square center: (0.4, -0.45), distance between opposite robots: 0.8
        # Each robot is 0.4 away from center

        # North arm (North, +Y direction)
        self.scene.north_arm.init_state.pos = (0.4, -0.05, 0.01)
        self.scene.north_arm.init_state.rot = euler_deg_to_quat(0, 0, 0)  # 0° yaw, facing inward

        # South arm (South, -Y direction, opposite of North)
        self.scene.south_arm.init_state.pos = (0.4, -0.85, 0.01)
        self.scene.south_arm.init_state.rot = euler_deg_to_quat(0, 0, 180)  # 180° yaw, facing inward

        # East arm (East, +X direction)
        self.scene.east_arm.init_state.pos = (0.8, -0.45, 0.01)
        self.scene.east_arm.init_state.rot = euler_deg_to_quat(0, 0, -90)  # -90° yaw, facing inward

        # West arm (West, -X direction, opposite of East)
        self.scene.west_arm.init_state.pos = (0.0, -0.45, 0.01)
        self.scene.west_arm.init_state.rot = euler_deg_to_quat(0, 0, 90)  # 90° yaw, facing inward

        # Desk position at center of square
        self.scene.desk.init_state.pos = (0.4, -0.45, 0.05)
        self.scene.desk.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        parse_usd_and_create_subassets(DESK_LIFT_USD_PATH, self)

    def use_teleop_device(self, teleop_device) -> None:
        self.actions = init_action_cfg(self.actions, device=teleop_device)
        if teleop_device == "keyboard":
            self.scene.robot.spawn.rigid_props.disable_gravity = True

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

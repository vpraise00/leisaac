import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from leisaac.assets.robots.lerobot import SO101_FOLLOWER_CFG
from leisaac.assets.scenes.collaborate_demo import (
    COLLABORATE_DEMO_CFG,
    COLLABORATE_DEMO_USD_PATH,
    MINI_TABLE_CFG,
)
from leisaac.tasks.lift_desk_quad_arm.lift_desk_quad_arm_env_cfg import (
    EventCfg as LiftDeskEventCfg,
    LiftDeskQuadArmEnvCfg,
    LiftDeskQuadArmSceneCfg,
    euler_deg_to_quat,
)


@configclass
class QuadArmCollaborateDemoSceneCfg(LiftDeskQuadArmSceneCfg):
    """Scene variant used for collaboration demos with custom top cameras."""

    scene: AssetBaseCfg = COLLABORATE_DEMO_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    desk: AssetBaseCfg | None = None
    mini_table: RigidObjectCfg = MINI_TABLE_CFG

    north_arm = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/North_Robot")
    east_arm = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/East_Robot")
    west_arm = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/West_Robot")
    south_arm = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/South_Robot")

    # Boom-mounted top cameras angled toward the workspace center (0, 0, 0.745).
    south_top: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/SouthTopCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.35, 0.0, 1.2),
            rot=(0.227692782, -0.669444544, -0.669444544, 0.227692782),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

    north_top: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/NorthTopCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.35, 0.0, 1.2),
            rot=(0.227692782, -0.669444544, 0.669444544, -0.227692782),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

@configclass
class QuadArmCollaborateDemoEventCfg(LiftDeskEventCfg):
    """Disable desk-specific reset events since the collaborate scene bakes the furniture."""

    reset_desk = None


@configclass
class QuadArmCollaborateDemoEnvCfg(LiftDeskQuadArmEnvCfg):
    """Environment configuration used by the quad arm collaboration demo."""

    scene: QuadArmCollaborateDemoSceneCfg = QuadArmCollaborateDemoSceneCfg(env_spacing=10.0)
    events: QuadArmCollaborateDemoEventCfg = QuadArmCollaborateDemoEventCfg()
    scene_asset_path: str = COLLABORATE_DEMO_USD_PATH

    def __post_init__(self) -> None:
        super().__post_init__()

        # Collaboration demos typically last longer than the lift-desk episodes.
        self.episode_length_s = 12.0

        # Wider framing to ensure all four arms plus the desk remain in view.
        self.viewer.eye = (-2.0, -2.4, 1.9)
        self.viewer.lookat = (0.4, -0.45, 0.4)

        # Position SO101 follower arms per provided warehouse layout (deg -> quaternion yaw).
        placements = [
            (self.scene.north_arm, (0.30, 0.325, 0.745), -45.0),
            (self.scene.east_arm, (0.325, -0.30, 0.745), -135.0),
            (self.scene.west_arm, (-0.30, -0.325, 0.745), 135.0),
            (self.scene.south_arm, (-0.325, 0.30, 0.745), 45.0),
        ]
        for arm_cfg, pos, yaw in placements:
            arm_cfg.init_state.pos = pos
            arm_cfg.init_state.rot = euler_deg_to_quat(0.0, 0.0, yaw)
            # Start in a tucked/sitting pose to avoid colliding with the table top.
            arm_cfg.init_state.joint_pos = {
                "shoulder_pan": 0.0,
                # Relaxed tucked pose (closer to initial waypoints to reduce IK jumps).
                "shoulder_lift": -0.9,
                "elbow_flex": 1.2,
                "wrist_flex": 0.8,
                "wrist_roll": 0.0,
                "gripper": 0.0,
            }

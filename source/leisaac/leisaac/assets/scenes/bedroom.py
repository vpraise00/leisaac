from pathlib import Path

from leisaac.utils.constant import ASSETS_ROOT

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg


"""Configuration for the Toy Room Scene"""
SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"

<<<<<<< Updated upstream:source/leisaac/leisaac/assets/scenes/bedroom.py
LIGHTWHEEL_BEDROOM_USD_PATH = str(SCENES_ROOT / "lightwheel_bedroom" / "scene.usd")
=======
DESK_LIFT_USD_PATH = str(SCENES_ROOT / "desk_lift" / "scene_empty_big.usd")
DESK_USD_PATH = str(SCENES_ROOT / "desk_lift" / "cl_desk.usd")
>>>>>>> Stashed changes:source/leisaac/leisaac/assets/scenes/desk_lift.py

LIGHTWHEEL_BEDROOM_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LIGHTWHEEL_BEDROOM_USD_PATH,
    )
)

DESK_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=DESK_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,  # Let gravity keep the desk seated until arms lift it
            max_depenetration_velocity=10.0,  # Allow faster separation when contacts compress
            kinematic_enabled=False,
            linear_damping=0.2,
            angular_damping=0.2,
        ),
        mass_props=sim_utils.MassPropertiesCfg(
            mass=50,  # 50g effective mass so desk stays light but non-zero
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.02,  # Tighter contact skin to reduce perceived penetration
            rest_offset=0.0,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(),
)

from pathlib import Path

from leisaac.utils.constant import ASSETS_ROOT

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg


"""Configuration for the Desk Lift Scene"""
SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"

DESK_LIFT_USD_PATH = str(SCENES_ROOT / "desk_lift" / "lab_table.usd")

DESK_LIFT_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=DESK_LIFT_USD_PATH,
    )
)

# # Desk configuration with collision and physics properties
# DESK_CFG = RigidObjectCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=str(SCENES_ROOT / "desk_lift" / "cl_desk.usd"),
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             max_depenetration_velocity=10.0,
#             kinematic_enabled=False,
#             linear_damping=0.5,  # Increased from 0.2 to reduce oscillation
#             angular_damping=0.5,  # Increased from 0.2 to reduce oscillation
#             max_linear_velocity=5.0,  # Limit velocity to prevent instability
#             max_angular_velocity=5.0,
#         ),
#         mass_props=sim_utils.MassPropertiesCfg(
#             mass=0.0001,  # 50g - very light desk
#         ),
#         collision_props=sim_utils.CollisionPropertiesCfg(
#             collision_enabled=True,
#             contact_offset=0.02,
#             rest_offset=0.0,
#         ),
#     ),
#     init_state=RigidObjectCfg.InitialStateCfg(
#         pos=(0.47, -0.58, 0.05),
#         rot=(1.0, 0.0, 0.0, 0.0),
#     ),
# )

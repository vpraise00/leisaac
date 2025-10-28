from pathlib import Path

from leisaac.utils.constant import ASSETS_ROOT

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg


"""Configuration for the Toy Room Scene"""
SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"

DESK_LIFT_USD_PATH = str(SCENES_ROOT / "desk_lift" / "scene.usd")

DESK_LIFT_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=DESK_LIFT_USD_PATH,
    )
)

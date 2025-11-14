from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

from leisaac.utils.constant import ASSETS_ROOT


SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"
COLLABORATE_DEMO_USD_PATH = str(SCENES_ROOT / "collaborate_demo" / "lab_table.usd")

COLLABORATE_DEMO_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=COLLABORATE_DEMO_USD_PATH,
    )
)

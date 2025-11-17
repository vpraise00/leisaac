from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

from leisaac.utils.constant import ASSETS_ROOT


SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"
COLLABORATE_DEMO_USD_PATH = str(SCENES_ROOT / "collaborate_demo" / "lab_table.usd")
MINI_TABLE_USD_PATH = str(SCENES_ROOT / "collaborate_demo" / "mini_table.usd")

COLLABORATE_DEMO_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=COLLABORATE_DEMO_USD_PATH,
    )
)

MINI_TABLE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/MiniTable",
    spawn=sim_utils.UsdFileCfg(
        usd_path=MINI_TABLE_USD_PATH,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

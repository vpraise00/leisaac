import gymnasium as gym

gym.register(
    id="LeIsaac-QuadArm-CollaborateDemo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quad_arm_collaborate_demo_env_cfg:QuadArmCollaborateDemoEnvCfg",
    },
)

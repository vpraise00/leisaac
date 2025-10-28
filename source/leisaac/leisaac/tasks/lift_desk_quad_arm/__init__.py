import gymnasium as gym

gym.register(
    id='LeIsaac-SO101-LiftDesk-QuadArm-v0',
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_desk_quad_arm_env_cfg:LiftDeskQuadArmEnvCfg",
    },
)

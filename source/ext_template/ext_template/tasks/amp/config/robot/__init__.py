import gymnasium as gym

from . import agents, env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Robot-AMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.RobotAmpEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.amp_rsl_rl_cfg:RobotAmpOnPolicyRunnerCfg",
    },
)


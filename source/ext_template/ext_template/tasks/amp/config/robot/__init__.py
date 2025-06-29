import gymnasium as gym

from . import agents, env_cfg

##
# Register Gym environments.
##


gym.register(
    id="Template-Isaac-AMP-H1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.H1AmpEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.amp_rsl_rl_cfg:H1AmpRunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-AMP-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.G1AmpEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.amp_rsl_rl_cfg:G1AmpRunnerCfg",
    },
)

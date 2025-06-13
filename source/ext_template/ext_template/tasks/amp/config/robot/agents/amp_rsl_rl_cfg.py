from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from dataclasses import MISSING
from pathlib import Path


@configclass
class DiscriminatorCfg:
    hidden_dims: list[int] = MISSING
    """The hidden dimensions of the discriminator network."""
    reward_scale: float = 0.5
    """style reward coefficient of the policy. default value : 0.5."""


@configclass
class RslRlAmpOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    amp_data_path: Path = MISSING
    """Directory containing the .npy motion files"""
    dataset_names: list[str] = MISSING
    """List of dataset filenames (no extension)"""
    dataset_weights: list[float] = MISSING
    """List of sampling weights (for minibatch sampling)"""
    slow_down_factor: int = 1
    """Integer factor to slow down original data"""


@configclass
class RobotAmpOnPolicyRunnerCfg(RslRlAmpOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 100
    experiment_name = "robot_amp"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 512],
        critic_hidden_dims=[512, 512],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    discriminator = DiscriminatorCfg(
        hidden_dims=[512, 512],
        reward_scale=0.5,
    )
    amp_data_path = MISSING
    dataset_names = MISSING
    dataset_weights = MISSING
    slow_down_factor = 1


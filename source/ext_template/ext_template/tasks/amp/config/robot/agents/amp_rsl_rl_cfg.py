from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from dataclasses import MISSING
from pathlib import Path

@configclass
class RslRlDiscriminatorCfg:
    """Configuration for the discriminator network."""

    hidden_dims: list[int] = MISSING
    """The hidden dimensions of the discriminator network."""
    
    reward_scale: float = 1.0
    """Style reward Scale of the policy. Default value : 1.0"""

@configclass
class RslRlAmpOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration of the runner for AMP on-policy algorithms."""
    
    discriminator: RslRlDiscriminatorCfg = MISSING
    
    amp_data_path: Path = MISSING
    """Directory containing the .npy motion files."""
    dataset_names: list[str] = MISSING
    """List of dataset filenames(no extension)."""
    
    dataset_weights: list[str] = MISSING
    """List of sampling weights(for minibatch sampling)."""
    
    slow_down_factor: int = 1
    """integer factor to slow down original data. Default value : 1"""
    
@configclass
class RslRlAmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the AMP-PPO algorithm."""
    
    
@configclass
class RobotAmpRunnerCfg(RslRlAmpOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 200
    experiment_name = "robot_amp"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims = [1024, 512],
        critic_hidden_dims = [1024, 512],
        activation="elu"
    )
    # algorithm=RslRlPpoAlgorithmCfg(
    #     class_name="AMP_PPO",
    #     value_loss_coef=1.0,
    #     use_clipped_value_loss=True,
    #     clip_param=0.2,
    #     entropy_coef=0.005,
    #     num_learning_epochs=5,
    #     num_mini_batches=4,
    #     learning_rate=1.0e-3,
    #     schedule="adaptive",
    #     gamma=0.99,
    #     lam=0.95,
    #     desired_kl=0.01,
    #     max_grad_norm=1.0,
    # )
    algorithm=RslRlPpoAlgorithmCfg(
        class_name="AMP_PPO",
        value_loss_coef=2.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=6,
        num_mini_batches=2,
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    discriminator = RslRlDiscriminatorCfg(
        hidden_dims = [1024,512],
        reward_scale = 2.0
    )
    
    amp_data_path = MISSING
    dataset_names = MISSING
    dataset_weights = MISSING
    slow_down_factor = 1
    
@configclass
class G1AmpRunnerCfg(RobotAmpRunnerCfg):
    experiment_name = "G1_AMP"
    amp_data_path = Path("/home/ysk/ws/isaaclab_amp_rsl_rl/motions/g1")
    dataset_names = ['walk_converted']
    dataset_weights = [1.0]
    
@configclass
class H1AmpRunnerCfg(RobotAmpRunnerCfg):
    experiment_name = "H1_AMP"
    amp_data_path = Path("/home/ysk/ws/isaaclab_amp_rsl_rl/motions/h1")
    dataset_names = ['walk_converted']
    dataset_weights = [1.0]
        

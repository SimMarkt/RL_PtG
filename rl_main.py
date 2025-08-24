"""
---------------------------------------------------------------------------------------------
RL_PtG: Deep Reinforcement Learning for Power-to-Gas Dispatch Optimization
GitHub Repository: https://github.com/SimMarkt/RL_PtG

rl_main:
> Main script for training deep reinforcement learning (RL) algorithms on the PtG-CH4 dispatch task.
> Adapts to different computational environments: 
    - a local personal computer ('pc') or a computing cluster with SLURM management ('slurm').
---------------------------------------------------------------------------------------------
"""

# pylint: disable=no-member, import-outside-toplevel

# ---------------------------------Import Python libraries-----------------------------------
import os
import torch as th

# Library for the RL environment
from gymnasium.envs.registration import registry, register

# Libraries with utility functions and classes
from src.rl_utils import load_data, initial_print, config_print, create_vec_envs
from src.rl_utils import Preprocessing, Postprocessing
from src.rl_config_agent import AgentConfiguration
from src.rl_config_env import EnvConfiguration
from src.rl_config_train import TrainConfiguration

def computational_resources(train_config: TrainConfiguration) -> None:
    """
        Configures computational resources and sets the random seed for the current thread
        :param train_config: Training configuration (class object)
    """
    print("Set computational resources...")
    train_config.path = os.path.dirname(__file__)
    if train_config.com_conf == 'pc':
        print("---Computation on local resources")
        train_config.seed_train = train_config.r_seed_train[0]
        train_config.seed_test = train_config.r_seed_test[0]
    else:
        print("---SLURM Task ID:", os.environ['SLURM_PROCID'])
        # Thread ID of the specific SLURM process in parallel computing on a computing cluster
        train_config.slurm_id = int(os.environ['SLURM_PROCID'])
        assert train_config.slurm_id <= len(train_config.r_seed_train), (
            f"No. of SLURM threads exceeds the No. of specified random seeds "
            f"({len(train_config.r_seed_train)}) - please add additional seed values to "
            "RL_PtG/config/config_train.yaml -> r_seed_train & r_seed_test"
            )
        train_config.seed_train = train_config.r_seed_train[train_config.slurm_id]
        train_config.seed_test = train_config.r_seed_test[train_config.slurm_id]
    if train_config.device == 'cpu':
        print("---Utilization of CPU\n")
    elif train_config.device == 'auto':
        print("---Automatic hardware utilization (GPU, if possible)\n")
    else:
        print("---CUDA available:", th.cuda.is_available(),
              "GPU device:", th.cuda.get_device_name(0), "\n")

def check_env(env_id):
    """
        Registers the Gymnasium environment if it is not already in the registry
        :param env_id: Unique identifier for the environment
    """
    if env_id not in registry:      # Check if the environment is already registered
        try:
            # Import the ptg_gym_env environment
            from env.ptg_gym_env import PTGEnv

            # Register the environment
            register(
                id=env_id,
                entry_point="env.ptg_gym_env:PTGEnv",  # Path to the environment class
            )
            print(f"---Environment '{env_id}' registered successfully!\n")
        except ImportError as e:
            print(f"Error importing the environment module: {e}")
        except Exception as e:
            print(f"Error registering the environment: {e}")
    else:
        print(f"---Environment '{env_id}' is already registered.\n")

def main():
    """
        Main function to set up and execute the RL training process.
    """
    # -----------------------------Initialize the RL configuration-------------------------------
    initial_print()
    agent_config = AgentConfiguration()
    env_config = EnvConfiguration()
    train_config = TrainConfiguration()
    computational_resources(train_config)
    str_id = config_print(agent_config, env_config, train_config)

    # --------------------------------------Preprocessing----------------------------------------
    print("Preprocessing...")
    dict_price_data, dict_op_data = load_data(env_config, train_config)

    # Initialize preprocessing with calculation of potential rewards and load identifiers
    preprocess = Preprocessing(dict_price_data, dict_op_data,
                               agent_config, env_config, train_config)
    # Create dictionaries for kwargs of training and test environments
    env_kwargs_data = {'env_kwargs_train': preprocess.dict_env_kwargs("train"),
                       'env_kwargs_val': preprocess.dict_env_kwargs("val"),
                       'env_kwargs_test': preprocess.dict_env_kwargs("test"),}

    # Instantiate the vectorized environments
    print("Load environment...")
    env_id = 'PtGEnv-v0'
    # Check the Gymnasium environment registry
    check_env(env_id)
    # Create vectorized environments
    env_train, env_test_post, eval_callback_val, eval_callback_test = create_vec_envs(
        env_id,
        str_id,
        agent_config,
        train_config,
        env_kwargs_data
    )
    # Set path for tensorboard data (for monitoring RL training)
    tb_log = "tensorboard/" + str_id

    # Set up the RL model with the specified algorithm
    if train_config.model_conf == "simple_train" or train_config.model_conf == "save_model":
        # Train RL model from scratch# Train RL model from scratch
        model = agent_config.set_model(env_train, tb_log, train_config)
    else:   # Load a pretrained model
        model = agent_config.load_model(env_train, tb_log,
                                        f"{train_config.path}{train_config.path_files}{str_id}",
                                        'train')

    # --------------------------------------RL Training------------------------------------------
    print("Training... >>>", str_id, "<<< \n")
    if train_config.val_n_test:
        # Evaluate the RL agent on both validation and test sets
        model.learn(total_timesteps=train_config.train_steps,
                    callback=[eval_callback_val, eval_callback_test])
    else:
        # Evaluate the RL agent only on the validation set
        model.learn(total_timesteps=train_config.train_steps,
                    callback=[eval_callback_val])
    print("...finished RL training\n")

    # ---------------------------------------Save model------------------------------------------
    if train_config.model_conf == "save_model" or train_config.model_conf == "save_load_model":
        print("Save RL agent under ./logs/ ... \n")
        agent_config.save_model(model)

    # -------------------------------------Post-processing---------------------------------------
    print("Postprocessing...")
    postprocess = Postprocessing(str_id, agent_config, env_config, train_config,
                                 env_test_post, preprocess)
    postprocess.test_performance()
    postprocess.plot_results()

if __name__ == '__main__':
    main()

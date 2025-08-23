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

# --------------------------------------------Import Python libraries---------------------------------------------
import os
import torch as th

# Library for the RL environment
from gymnasium.envs.registration import registry, register 

# Libraries with utility functions and classes
from src.rl_utils import load_data, initial_print, config_print, Preprocessing, Postprocessing, create_vec_envs#, create_vec_envs
from src.rl_config_agent import agent_configuration
from src.rl_config_env import env_configuration
from src.rl_config_train import train_configuration

def computational_resources(train_config):
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
        train_config.slurm_id = int(os.environ['SLURM_PROCID'])         # Thread ID of the specific SLURM process in parallel computing on a computing cluster
        assert train_config.slurm_id <= len(train_config.r_seed_train), f"No. of SLURM threads exceeds the No. of specified random seeds ({len(train_config.r_seed_train)}) - please add additional seed values to RL_PtG/config/config_train.yaml -> r_seed_train & r_seed_test"
        train_config.seed_train = train_config.r_seed_train[train_config.slurm_id]
        train_config.seed_test = train_config.r_seed_test[train_config.slurm_id]
    if train_config.device == 'cpu':    print("---Utilization of CPU\n")
    elif train_config.device == 'auto': print("---Automatic hardware utilization (GPU, if possible)\n")
    else:                       print("---CUDA available:", th.cuda.is_available(), "GPU device:", th.cuda.get_device_name(0), "\n")

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
    # --------------------------------------Initialize the RL configuration---------------------------------------
    initial_print()
    agent_config = agent_configuration()
    env_config = env_configuration()
    train_config = train_configuration()
    computational_resources(train_config)
    str_id = config_print(agent_config, env_config, train_config)
    
    # -----------------------------------------------Preprocessing------------------------------------------------
    print("Preprocessing...")
    dict_price_data, dict_op_data = load_data(env_config, train_config)

    # Initialize preprocessing with calculation of potential rewards and load identifiers
    Preprocess = Preprocessing(dict_price_data, dict_op_data, agent_config, env_config, train_config)    
    # Create dictionaries for kwargs of training and test environments
    env_kwargs_data = {'env_kwargs_train': Preprocess.dict_env_kwargs("train"),
                       'env_kwargs_val': Preprocess.dict_env_kwargs("val"),
                       'env_kwargs_test': Preprocess.dict_env_kwargs("test"),}

    # Instantiate the vectorized environments
    print("Load environment...")
    env_id = 'PtGEnv-v0'
    check_env(env_id)                                                                                                   # Check the Gymnasium environment registry
    env_train, env_test_post, eval_callback_val, eval_callback_test = create_vec_envs(env_id, str_id, agent_config, train_config, env_kwargs_data)          # Create vectorized environments
    tb_log = "tensorboard/" + str_id                                                                                    # Set path for tensorboard data (for monitoring RL training) 

    # Set up the RL model with the specified algorithm
    if train_config.model_conf == "simple_train" or train_config.model_conf == "save_model":          # Train RL model from scratch
        model = agent_config.set_model(env_train, tb_log, train_config)
    else:                                                                                           # Load a pretrained model
        model = agent_config.load_model(env_train, tb_log, f"{train_config.path}{train_config.path_files}{str_id}", 'train')

    # ------------------------------------------------RL Training-------------------------------------------------
    print("Training... >>>", str_id, "<<< \n")
    if train_config.val_n_test:  model.learn(total_timesteps=train_config.train_steps, callback=[eval_callback_val, eval_callback_test])  # Evaluate the RL agent on both validation and test sets
    else:                       model.learn(total_timesteps=train_config.train_steps, callback=[eval_callback_val])                      # Evaluate the RL agent only on the validation set
    print("...finished RL training\n")

    # ------------------------------------------------Save model--------------------------------------------------
    if train_config.model_conf == "save_model" or train_config.model_conf == "save_load_model":
        print("Save RL agent under ./logs/ ... \n") 
        agent_config.save_model(model)
    
    # ----------------------------------------------Post-processing-----------------------------------------------
    print("Postprocessing...")
    PostProcess = Postprocessing(str_id, agent_config, env_config, train_config, env_test_post, Preprocess)
    PostProcess.test_performance()
    PostProcess.plot_results()

if __name__ == '__main__':
    main()




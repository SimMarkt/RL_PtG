# ----------------------------------------------------------------------------------------------------------------
# RL_PtG: Deep Reinforcement Learning for Power-to-Gas dispatch optimization
# https://github.com/SimMarkt/RL_PtG

# rl_main: 
# > Main programming script for training of deep RL algorithms on the PtG-CH4 dispatch task 
# > Distinguishes between present computational resources: local personal computer ('pc') or computing cluster with SLURM management ('slurm')
# ----------------------------------------------------------------------------------------------------------------

# --------------------------------------------Import Python libraries---------------------------------------------
# Standard libraries
import os
import torch as th

# Libraries for gymnasium and stable_baselines
from gymnasium.envs.registration import registry, register, make
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# Libraries with utility functions and classes
from src.rl_utils import load_data, initial_print, config_print, Preprocessing, Postprocessing, create_vec_envs#, create_vec_envs
from src.rl_config_agent import AgentConfiguration
from src.rl_config_env import EnvConfiguration
from src.rl_config_train import TrainConfiguration

################TODO: ENERGIEMARKT DATEN ANPASSEN###############################################################

def computational_resources(TrainConfig):
    """
        Set computational resources and the random seed of the present thread
        :param TrainConfig: Training configuration in a class object
    """
    print("Set computational resources...")
    TrainConfig.path = os.path.dirname(__file__)
    if TrainConfig.com_conf == 'pc': 
        print("---Computation on local resources")
        TrainConfig.seed_train = TrainConfig.r_seed_train[0]
        TrainConfig.seed_test = TrainConfig.r_seed_test[0]
    else: 
        print("---SLURM Task ID:", os.environ['SLURM_PROCID'])
        TrainConfig.slurm_id = int(os.environ['SLURM_PROCID'])         # Thread ID of the specific SLURM process in parallel computing on a cluster
        assert TrainConfig.slurm_id <= len(TrainConfig.r_seed_train), f"No. of SLURM threads exceeds the No. of specified random seeds ({len(TrainConfig.r_seed_train)}) - please add additional seed values to RL_PtG/config/config_train.yaml -> r_seed_train & r_seed_test"
        TrainConfig.seed_train = TrainConfig.r_seed_train[TrainConfig.slurm_id]
        TrainConfig.seed_test = TrainConfig.r_seed_test[TrainConfig.slurm_id]
    if TrainConfig.device == 'cpu':    print("---Utilization of CPU\n")
    elif TrainConfig.device == 'auto': print("---Automatic hardware utilization (GPU, if possible)\n")
    else:                       print("---CUDA available:", th.cuda.is_available(), "GPU device:", th.cuda.get_device_name(0), "\n")

def check_env(env_id):
    """
        Adds the Gymnasium environment to the registry if is not already registered
        :param env_id: ID of the environment
    """
    if env_id not in registry:      # Check if the environment is already registered
        try:
            # Import your custom environment class
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


# def create_vec_envs(env_id, str_id, AgentConfig, TrainConfig, env_kwargs_data):            ######### MAKE SHORTER ########### MAYBE WITH A DECORATOR ################
#     """
#         Creates vectorized environments for training, validation, and testing
#         :param env_id: ID of the environment
#         :param str_id: String for identification of the present training run
#         :param AgentConfig: Agent configuration in a class object
#         :param TrainConfig: Training configuration in a class object
#         :return env_kwargs_data: Dictionaries with kwargs of training, validation, and test environments
#         :return eval_callback_val: Callback object with the validation environment for periodic evaluation (validation for hyperparameter tuning)
#         :return eval_callback_test: Callback object with the test environment for periodic evaluation (testing for error evaluation of the tuned RL model)
#     """

#     # Training environment
#     if TrainConfig.parallel == "Singleprocessing":
#         # DummyVecEnv -> computes each workers interaction in serial, if calculating the env itself is quite fast
#         env_train = make_vec_env(env_id=env_id, n_envs=AgentConfig.n_envs, seed=TrainConfig.seed_train, vec_env_cls=DummyVecEnv,
#                            env_kwargs=dict(dict_input=env_kwargs_data['env_kwargs_train'], train_or_eval=TrainConfig.train_or_eval,
#                                            render_mode="None"))
#     elif TrainConfig.parallel == "Multiprocessing":
#         # SubprocVecEnv for multiprocessing -> computes each workers interaction in parallel, if calculating the env itself is quite slow
#         env_train = make_vec_env(env_id=env_id, n_envs=AgentConfig.n_envs, seed=TrainConfig.seed_train, vec_env_cls=SubprocVecEnv,
#                            env_kwargs=dict(dict_input=env_kwargs_data['env_kwargs_train'], train_or_eval=TrainConfig.train_or_eval,
#                                            render_mode="None"))
#                          #, vec_env_kwargs=dict(start_method="fork"/"spawn"/"forkserver")) # optional
#     else:
#         assert False, 'Choose either "Singleprocessing" or "Multiprocessing" in RL_PTG/config/config_train.yaml -> parallel!'

#     env_train = VecNormalize(env_train, norm_obs=False)   # Normalization of rewards with a moving average (norm_obs=False -> observations are normalized separately within the environment)

#     # Environment for validation, EvalCallback creates a callback function which is called during RL learning ###################ZUSAMMENFASSEN ENV_VAL UND ENV_TEST
#     env_val = make_vec_env(env_id, n_envs=TrainConfig.eval_trials, seed=TrainConfig.seed_test, vec_env_cls=DummyVecEnv,
#                             env_kwargs=dict(dict_input=env_kwargs_data['env_kwargs_val'], train_or_eval=TrainConfig.train_or_eval,
#                                             render_mode="None"))
#     env_val = VecNormalize(env_val, norm_obs=False)

#     eval_callback_val = EvalCallback(env_val,
#                                       best_model_save_path= TrainConfig.path + "/logs/" + str_id + "_val/",
#                                       n_eval_episodes=TrainConfig.eval_trials,
#                                       log_path=TrainConfig.path + "/logs/", 
#                                       eval_freq=int(TrainConfig.test_steps / AgentConfig.n_envs),              # eval_freq=int(TrainConfig.test_steps / EnvConfig.n_envs) performs validation after test_steps steps independent of the No. of workers 
#                                       deterministic=True, render=False, verbose=0)

#     # Environment for testing, EvalCallback creates a callback function which is called during RL learning
#     env_test = make_vec_env(env_id, n_envs=TrainConfig.eval_trials, seed=TrainConfig.seed_test, vec_env_cls=DummyVecEnv,
#                             env_kwargs=dict(dict_input=env_kwargs_data['env_kwargs_test'], train_or_eval=TrainConfig.train_or_eval,
#                                             render_mode="None"))
#     env_test = VecNormalize(env_test, norm_obs=False)

#     eval_callback_test = EvalCallback(env_test,
#                                       best_model_save_path=TrainConfig.path + "/logs/" + str_id + "_test/",
#                                       n_eval_episodes=TrainConfig.eval_trials,
#                                       log_path=TrainConfig.path + "/logs/", 
#                                       eval_freq=int(TrainConfig.test_steps / AgentConfig.n_envs),              # eval_freq=int(test_steps / EnvConfig.n_envs) performs validation after test_steps steps independent of the No. of workers 
#                                       deterministic=True, render=False, verbose=0)
    
#     return env_train, env_test, eval_callback_val, eval_callback_test


def main():
    # ----------------------------------------Initialize RL configuration-----------------------------------------
    initial_print()
    AgentConfig = AgentConfiguration()
    EnvConfig = EnvConfiguration()
    TrainConfig = TrainConfiguration()
    computational_resources(TrainConfig)
    str_id = config_print(AgentConfig, EnvConfig, TrainConfig)
    
    # -----------------------------------------------Preprocessing------------------------------------------------
    print("Preprocessing...")
    dict_price_data, dict_op_data = load_data(EnvConfig, TrainConfig)

    # Initialize preprocessing with calculation of potential rewards and load identifiers
    Preprocess = Preprocessing(dict_price_data, dict_op_data, AgentConfig, EnvConfig, TrainConfig)    
    # Create dictionaries for kwargs of train and test environments
    env_kwargs_data = {'env_kwargs_train': Preprocess.dict_env_kwargs("train"),
                       'env_kwargs_val': Preprocess.dict_env_kwargs("val"),
                       'env_kwargs_test': Preprocess.dict_env_kwargs("test"),}

    # Instantiate the vectorized environments
    print("Load environment...")
    env_id = 'PtGEnv-v0'
    check_env(env_id)                                                                                                   # Check the Gymnasium environment registry
    env_train, env_test, eval_callback_val, eval_callback_test = create_vec_envs(env_id, str_id, AgentConfig, TrainConfig, env_kwargs_data)          # Create vectorized environments
    tb_log = "tensorboard/" + str_id                                                                                    # Set path for tensorboard data (for monitoring RL training) 

    # Set up the RL model with the specified algorithm
    if TrainConfig.model_conf == "simple_train" or TrainConfig.model_conf == "save_model":          # Train RL model from scratch
        model = AgentConfig.set_model(env_train, tb_log, TrainConfig)
    else:                                                                                           # Load a pretrained model
        model = AgentConfig.load_model(env_train, tb_log, str_id)

    # ------------------------------------------------RL Training-------------------------------------------------
    print("Training... >>>", str_id, "<<< \n")
    if TrainConfig.val_n_test:  model.learn(total_timesteps=TrainConfig.train_steps, callback=[eval_callback_val, eval_callback_test])  # Evaluate the RL agent on both validation and test sets
    else:                       model.learn(total_timesteps=TrainConfig.train_steps, callback=[eval_callback_val])                      # Evaluate the RL agent only on the validation set
    print("...finished RL training\n")

    # ------------------------------------------------Save model--------------------------------------------------
    if TrainConfig.model_conf == "save_model" or TrainConfig.model_conf == "save_load_model":
        print("Save RL agent under ./logs/ ... \n") 
        AgentConfig.save_model(model)
    
    # ----------------------------------------------Postprocessing------------------------------------------------
    print("Postprocessing...")
    PostProcess = Postprocessing(str_id, dict_price_data, dict_op_data, AgentConfig, EnvConfig, TrainConfig, env_test)
    PostProcess.test_performance()
    PostProcess.plot_results()

if __name__ == '__main__':
    main()




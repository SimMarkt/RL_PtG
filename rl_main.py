# ----------------------------------------------------------------------------------------------------------------
# RL_PtG: Deep Reinforcement Learning for Power-to-Gas dispatch optimization
# https://github.com/SimMarkt/RL_PtG

# rl_main_training: 
# > Contains the main script for training and evaluation of RL agents on PtG dispatch optimization
# ----------------------------------------------------------------------------------------------------------------

print("Import libraries...")
# standard libraries
import torch as th
import numpy as np
import yaml

# libraries for gym and stable_baselines
# from stable_baselines3 import PPO                 # Import algortihm of choice
from sb3_contrib import TQC
from gymnasium.envs.registration import registry, register, make
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# libraries with utility functions
from src.rl_utils import load_data, preprocessing_rew, preprocessing_array, define_episodes, dict_env_kwargs
from src.rl_glo_param import GlobalParams
from src.rl_hyp_param import HypParams, get_param_str

# ----------------------------------------------------------------------------------------------------------------------
print("Check for SLURM ID and CUDA availability...")
# print("---SLURM Task ID:", os.environ['SLURM_PROCID'])
# print("---CUDA available:", th.cuda.is_available(), "Device:", th.cuda.get_device_name(0))
device = 'auto'             # stablebaselines models with 'auto' will automatically choose gpus if available,
# device = 'cpu'
# device = 'cuda'

# ----------------------------------------------------------------------------------------------------------------------
print("Load setup and global parameters...")

GLOBAL_PARAMS = GlobalParams()
HYPER_PARAMS = HypParams()
str_vary = "TQC_hpc_62"           # signifies Rl algorithm for result labels
str_vary = str_vary + "_S" + str(GLOBAL_PARAMS.scenario)
str_vary = str_vary + "_" + str(GLOBAL_PARAMS.operation)
str_vary_former = HYPER_PARAMS.str_vary_former

if __name__ == '__main__':
    # Specify seeds
    # seed_train = HYPER_PARAMS.r_seed_train[int(os.environ['SLURM_PROCID'])]
    # seed_test = HYPER_PARAMS.r_seed_test[int(os.environ['SLURM_PROCID'])]

    # Instantiate the vectorized envs for parallelization
    print("Load environment...")
    env_id = 'PtGEnv-v0'

    # Check if the environment is already registered
    if env_id not in registry:
        try:
            # Import your custom environment class
            from env.ptg_gym_env import PTGEnv

            # Register the environment
            register(
                id=env_id,
                entry_point="env.ptg_gym_env:PTGEnv",  # Path to the environment class
            )
            print(f"Environment '{env_id}' registered successfully!")
        except ImportError as e:
            print(f"Error importing the environment module: {e}")
        except Exception as e:
            print(f"Error registering the environment: {e}")
    else:
        print(f"Environment '{env_id}' is already registered.")

    # Load configuration files
    with open("config/config_env.yaml", "r") as env_file:
        env_config = yaml.safe_load(env_file)
    
    with open("config/config_agent.yaml", "r") as agent_file:
        agent_config = yaml.safe_load(agent_file)
    
    print(env_config)
    print(agent_config)

    seed_train = HYPER_PARAMS.r_seed_train[0]
    seed_test = HYPER_PARAMS.r_seed_test[0]

    print("------- Calculate", str_vary, "---------")
    # load data
    print("Load data...")
    dict_price_data, dict_op_data = load_data()
    # data preprocessing: calculate potential reward and boolean identifier
    print("Preprocessing...")
    dict_pot_r_b, r_level = preprocessing_rew(dict_price_data)
    # data preprocessing: transform day-ahead datasets into np.arrays for calculation purposes
    e_r_b_train, e_r_b_cv, e_r_b_test, g_e_train, g_e_cv, g_e_test = preprocessing_array(dict_price_data, dict_pot_r_b)
    # define episodes and indices for choosing subsets of the training set randomly
    eps_sim_steps_train, eps_sim_steps_cv, eps_sim_steps_test, eps_ind, total_n_steps, n_eps_loops = define_episodes(dict_price_data, seed_train)
    # Create dictionaries for kwargs of train and test environment
    env_kwargs_train = dict_env_kwargs(eps_ind, e_r_b_train, g_e_train, dict_op_data, eps_sim_steps_train, n_eps_loops, r_level, "train")
    eps_ind_test = np.zeros(len(eps_ind), dtype=int)
    env_kwargs_cv = dict_env_kwargs(eps_ind_test, e_r_b_cv, g_e_cv, dict_op_data, eps_sim_steps_cv, n_eps_loops, r_level, "cv_test")
    env_kwargs_test = dict_env_kwargs(eps_ind_test, e_r_b_test, g_e_test, dict_op_data, eps_sim_steps_test, n_eps_loops, r_level, "cv_test")

    env_id = 'PtGEnv-v0'

    if GLOBAL_PARAMS.parallel == "Singleprocessing":
        # DummyVecEnv -> in serial, if calculating the env itself is quite fast (Multi-threading - OpenMP)
        env = make_vec_env(env_id=env_id, n_envs=GLOBAL_PARAMS.n_envs, seed=seed_train, vec_env_cls=DummyVecEnv,
                           env_kwargs=dict(dict_input=env_kwargs_train, train_or_eval=GLOBAL_PARAMS.train_or_eval,
                                           render_mode="None"))
    elif GLOBAL_PARAMS.parallel == "Multiprocessing":
        # SubprocVecEnv for multiprocessing -> in parallel, if calculating the env itself is quite slow (Multi-processing - MPI)
        env = make_vec_env(env_id=env_id, n_envs=GLOBAL_PARAMS.n_envs, seed=seed_train, vec_env_cls=SubprocVecEnv,
                           env_kwargs=dict(dict_input=env_kwargs_train, train_or_eval=GLOBAL_PARAMS.train_or_eval,
                                           render_mode="None"))
                         #, vec_env_kwargs=dict(start_method="fork"/"spawn"/"forkserver"))
    else:
        assert False, 'Choose either "Singleprocessing" or "Multiprocessing" for GLOBAL_PARAMS.parallel!'

    env = VecNormalize(env, norm_obs=False)   #### Change with inner env standardized reward using global reward regions from dict_pot_r_b_train

    # environment for cross-validation
    env_cv = make_vec_env(env_id, n_envs=5, seed=seed_test, vec_env_cls=DummyVecEnv,
                            env_kwargs=dict(dict_input=env_kwargs_cv, train_or_eval=GLOBAL_PARAMS.train_or_eval,
                                            render_mode="None"))

    env_cv = VecNormalize(env_cv, norm_obs=False)

    # Recommended: n_eval_episodes = 5 - 20, eval_freq = 50
    eval_callback_cv = EvalCallback(env_cv,
                                    best_model_save_path="./logs/" + str_vary + "_" + str(seed_train) + "/",
                                    n_eval_episodes=5,
                                    log_path="./logs/", eval_freq=int(80000 / GLOBAL_PARAMS.n_envs),     # 80000
                                    deterministic=True, render=False, verbose=0)

    # environment for testing
    env_test = make_vec_env(env_id, n_envs=5, seed=seed_test, vec_env_cls=DummyVecEnv,
                            env_kwargs=dict(dict_input=env_kwargs_test, train_or_eval=GLOBAL_PARAMS.train_or_eval,
                                            render_mode="None"))

    env_test = VecNormalize(env_test, norm_obs=False)

    # Recommended: n_eval_episodes = 5 - 20, eval_freq = 50
    eval_callback_test = EvalCallback(env_test,
                                      best_model_save_path="./logs/" + str_vary + "_" + str(seed_train) + "/",
                                      n_eval_episodes=5,
                                      log_path="./logs/", eval_freq=int(82000 / GLOBAL_PARAMS.n_envs),  # 100000
                                      deterministic=True, render=False, verbose=0)

    # ------------------------------------------------------------------------------------------------------------------
    print("Define model...")

    str_params_short, str_params_long = get_param_str(eps_sim_steps_train, seed_train)

    tb_log = "tensorboard/TQC/" + str_vary + str_params_short

    # Custom actor (pi) and value function (vf) networks with
    # same architecture net_arch with activation function (th.nn.ReLU, th.nn.Tanh)
    # Note: an extra linear layer will be added on top of the pi and
    # the vf nets, respectively
    if HYPER_PARAMS.activation == 0:
        activation_fn = th.nn.ReLU
    else:
        activation_fn = th.nn.Tanh
    net_arch = np.ones((HYPER_PARAMS.hidden_layers,), int) * HYPER_PARAMS.hidden_units
    net_arch = net_arch.tolist()
    policy_kwargs = dict(activation_fn=activation_fn, net_arch=net_arch,
                         n_critics=int(HYPER_PARAMS.n_critics), n_quantiles=int(HYPER_PARAMS.n_quantiles))

    if HYPER_PARAMS.gSDE == 0:
        gSDE = False
    else:
        gSDE = True

    if HYPER_PARAMS.ent_coeff == 2:
        ent_coeff = 'auto'
    else:
        ent_coeff = HYPER_PARAMS.ent_coeff

    if HYPER_PARAMS.model_conf != "load_model" or HYPER_PARAMS.model_conf != "save_load_model":
         model = TQC(
             "MultiInputPolicy",
             env,
             verbose=0,
             tensorboard_log=tb_log,
             top_quantiles_to_drop_per_net=int(HYPER_PARAMS.top_quantiles_drop),
             learning_rate=HYPER_PARAMS.alpha,
             gamma=HYPER_PARAMS.gamma,
             buffer_size=int(HYPER_PARAMS.buffer_size),
             batch_size=int(HYPER_PARAMS.batch_size),
             learning_starts=int(HYPER_PARAMS.learning_starts),
             tau=HYPER_PARAMS.tau,
             train_freq=HYPER_PARAMS.train_freq,
             ent_coef=ent_coeff,
             policy_kwargs=policy_kwargs,
             use_sde=gSDE,
             device=device,
             target_update_interval=HYPER_PARAMS.train_freq,
             seed=seed_train,
    )
    else:
        # # # Use the latest model for further training
        model = TQC.load("./logs/final/" + str_vary_former + "_sd" + str(seed_train) + "_step" + str(GLOBAL_PARAMS.initial_n_steps), tensorboard_log=tb_log)
        print(f"The loaded_model has {model.replay_buffer.size()} transitions in its buffer")
        model.load_replay_buffer("./logs/final/" + str_vary_former + "_sd" + str(seed_train) + "_step" + str(GLOBAL_PARAMS.initial_n_steps))
        print(f"The loaded_model has {model.replay_buffer.size()} transitions in its buffer")
        model.set_env(env)

    # ------------------------------------------------------------------------------------------------------------------
    print("Training:" + str_params_long)

    # checkpoint_callback = CheckpointCallback(save_freq=1e5, save_path='models/checkpoints/')
    # model.learn(total_timesteps=total_n_steps)  # , callback=checkpoint_callback)
    # model.learn(total_timesteps=total_n_steps, callback=[eval_callback_cv, eval_callback_test])  # , callback=checkpoint_callback)
    model.learn(total_timesteps=(GLOBAL_PARAMS.total_steps - GLOBAL_PARAMS.initial_n_steps), callback=[eval_callback_cv])
    # model.learn(total_timesteps=total_n_steps, progress_bar=True, callback=[TensorboardCallback(), eval_callback])  # , callback=checkpoint_callback)
    # model.learn(total_timesteps=total_n_steps, progress_bar=True, callback=TensorboardCallback())  # , callback=checkpoint_callback)

    if HYPER_PARAMS.model_conf == "save_model" or HYPER_PARAMS.model_conf == "save_load_model":
        print("Saving Final Model")
        model.save("./logs/final/" + str_vary + "_sd" + str(seed_train) + "_step" + str(GLOBAL_PARAMS.total_steps))
        model.save_replay_buffer("./logs/final/" + str_vary + "_sd" + str(seed_train) + "_step" + str(GLOBAL_PARAMS.total_steps))



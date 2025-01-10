# ----------------------------------------------------------------------------------------------------------------
# RL_PtG: Deep Reinforcement Learning for Power-to-Gas dispatch optimization
# https://github.com/SimMarkt/RL_PtG

# rl_param_agent: 
# > Contains hyperparameters of the RL agents
# > Converts the config_agent.yaml data into a class object for further processing
# > Contains three additional functions to specify, load, and save the Stable-Baselines3 models
# ----------------------------------------------------------------------------------------------------------------

import numpy as np
import torch as th
import yaml
from src.rl_param_env import EnvParams


class AgentParams:
    def __init__(self):
        # Load the environment configuration
        with open("config/config_agent.yaml", "r") as env_file:
            agent_config = yaml.safe_load(env_file)

        self.n_envs = agent_config['n_envs']            # Number environments/workers for training
        assert agent_config['rl_alg'] in agent_config['hyperparameters'], f'Wrong algorithm specified - data/config_agent.yaml -> 
                                                                            model_conf : {agent_config['rl_alg']} must match {agent_config['hyperparameters'].keys()}'
        self.rl_alg = agent_config['rl_alg']            # selected RL algorithm - already implemented [DQN, A2C, PPO, TD3, SAC, TQC]
        self.rl_alg_hyp = agent_config['hyperparameters'][self.rl_alg]     # hyperparameters of the algorithm
        

    def set_model(self, env, tb_log):
        """
            Specify the Stable-Baselines3 model for RL training
            :param env: environment
            :param tb_log: # tensorboard log file
            :device: device for computation (CPU or GPU)
            :seed_train: random seed for training
            :return model: Stable-Baselines3 model for RL training
        """
        
        # Implement the neural network architecture:
        #   Custom actor (pi) and value function (vf) networks with the
        #   same architecture net_arch and activation function (th.nn.ReLU, th.nn.Tanh)
        #   Note: an extra linear layer will be added on top of the pi and vf nets
        if self.rl_alg_hyp.activation == 0:
            activation_fn = th.nn.ReLU
        else:
            activation_fn = th.nn.Tanh
        net_arch = np.ones((self.rl_alg_hyp['hidden_layers'],), int) * self.rl_alg_hyp['hidden_units']
        net_arch = net_arch.tolist()

        # Set RL algorithms and specify hyperparameters
        if self.rl_alg == 'DQN':
            from stable_baselines3 import DQN           # import algorithm
            policy_kwargs = dict(activation_fn=activation_fn, net_arch=net_arch)
            model = DQN(
                "MultiInputPolicy",                                                     # policy type
                env,                                                                    # environment
                verbose=0,                                                              # without printing training details to the TUI
                tensorboard_log=tb_log,                                                 # tensorboard log file
                learning_rate=self.rl_alg_hyp['alpha'],                                 # learning rate
                gamma=self.rl_alg_hyp['gamma'],                                         # discount factor
                buffer_size=int(self.rl_alg_hyp['buffer_size']),                        # replay buffer size
                batch_size=int(self.rl_alg_hyp['batch_size']),                          # batch size
                exploration_initial_eps=self.rl_alg_hyp['eps_init'],                    # initial exploration coefficient
                exploration_final_eps=self.rl_alg_hyp['eps_fin'],                       # final exploration coefficient
                exploration_fraction=self.rl_alg_hyp['eps_fra'],                        # ratio of the training set for exploration annealing
                learning_starts=int(self.rl_alg_hyp['learning_starts']),                # No. of steps before starting the training of actor/critic networks
                tau=self.rl_alg_hyp['tau'],                                             # soft update parameter
                train_freq=self.rl_alg_hyp['train_freq'],                               # training frequency
                policy_kwargs=policy_kwargs,                                            # contains network hyperparameters
                device=self.device,                                                     # CPU or GPU
                target_update_interval=self.rl_alg_hyp['target_update_interval'],       # target network update interval
                seed=self.r_seed_train,                                                 # random seed
            )

        elif self.rl_alg == 'A2C':                      # import algorithm
            from stable_baselines3 import A2C
            policy_kwargs = dict(activation_fn=activation_fn, net_arch=net_arch)
            if self.rl_alg_hyp['gSDE'] == 0:
                gSDE = False
            else:
                gSDE = True
            if self.rl_alg_hyp['normalize_advantage'] == 0:
                normalize_advantage = False
            else:
                normalize_advantage = True
            model = A2C(
                "MultiInputPolicy",                                                     # policy type
                env,                                                                    # environment
                verbose=0,                                                              # without printing training details to the TUI
                tensorboard_log=tb_log,                                                 # tensorboard log file
                learning_rate=self.rl_alg_hyp['alpha'],                                 # learning rate
                gamma=self.rl_alg_hyp['gamma'],                                         # discount factor
                n_steps=self.rl_alg_hyp['n_steps'],                                     # No. of steps of the n-step TD update
                gae_lambda=self.rl_alg_hyp['gae_lambda'],                               # factor for generalized advantage estimation
                normalize_advantage=normalize_advantage,                                # normalize advantage (0: Off, 1: On)
                ent_coef=self.rl_alg_hyp['ent_coeff'],                                  # entropy coefficient
                policy_kwargs=policy_kwargs,                                            # contains network hyperparameters
                use_sde=gSDE,                                                           # gSDE exploration  (0: False, 1: True)
                device=self.device,                                                     # CPU or GPU
                seed=self.r_seed_train,                                                 # random seed
            )

        elif self.rl_alg == 'PPO':                      # import algorithm
            from stable_baselines3 import PPO
            policy_kwargs = dict(activation_fn=activation_fn, net_arch=net_arch)
            n_steps = int(self.rl_alg_hyp.n_steps_f * self.rl_alg_hyp.batch_size)       # number of steps of the n-step TD update
            if self.rl_alg_hyp['gSDE'] == 0:
                gSDE = False
            else:
                gSDE = True
            if self.rl_alg_hyp['normalize_advantage'] == 0:
                normalize_advantage = False
            else:
                normalize_advantage = True
            model = PPO(
                "MultiInputPolicy",                                                     # policy type
                env,                                                                    # environment
                verbose=0,                                                              # without printing training details to the TUI
                tensorboard_log=tb_log,                                                 # tensorboard log file
                learning_rate=self.rl_alg_hyp['alpha'],                                 # learning rate
                gamma=self.rl_alg_hyp['gamma'],                                         # discount factor
                batch_size=int(self.rl_alg_hyp['batch_size']),                          # batch size
                n_steps=n_steps,                                                        # No. of steps of the n-step TD update
                gae_lambda=self.rl_alg_hyp['gae_lambda'],                               # factor for generalized advantage estimation
                n_epochs=int(self.rl_alg_hyp['n_epoch']),                               # No. of epochs for mini batch training
                normalize_advantage=normalize_advantage,                                # normalize advantage (0: Off, 1: On)
                ent_coef=self.rl_alg_hyp['ent_coeff'],                                  # entropy coefficient
                policy_kwargs=policy_kwargs,                                            # contains network hyperparameters
                use_sde=gSDE,                                                           # gSDE exploration  (0: False, 1: True)
                device=self.device,                                                     # CPU or GPU
                seed=self.r_seed_train,                                                 # random seed
            )

        elif self.rl_alg == 'TD3':                      # import algorithm
            from stable_baselines3 import TD3
            policy_kwargs = dict(activation_fn=activation_fn, net_arch=net_arch)
            model = TD3(
                "MultiInputPolicy",                                                     # policy type
                env,                                                                    # environment
                verbose=0,                                                              # without printing training details to the TUI
                tensorboard_log=tb_log,                                                 # tensorboard log file
                learning_rate=self.rl_alg_hyp['alpha'],                                 # learning rate
                gamma=self.rl_alg_hyp['gamma'],                                         # discount factor
                buffer_size=int(self.rl_alg_hyp['buffer_size']),                        # replay buffer size
                batch_size=int(self.rl_alg_hyp['batch_size']),                          # batch size
                learning_starts=int(self.rl_alg_hyp['learning_starts']),                # No. of steps before starting the training of actor/critic networks
                tau=self.rl_alg_hyp['tau'],                                             # soft update parameter
                train_freq=self.rl_alg_hyp['train_freq'],                               # training frequency
                target_policy_noise=self.rl_alg_hyp['sigma_exp'],                       # standard deviation of Gaussian noise added to the target policy
                policy_kwargs=policy_kwargs,                                            # contains network hyperparameters
                device=self.device,                                                     # CPU or GPU
                seed=self.r_seed_train,                                                 # random seed
            )
        
        elif self.rl_alg == 'SAC':                      # import algorithm
            from stable_baselines3 import SAC
            policy_kwargs = dict(activation_fn=activation_fn, net_arch=net_arch)
            if self.rl_alg_hyp['gSDE'] == 0:
                gSDE = False
            else:
                gSDE = True
            model = SAC(
                "MultiInputPolicy",                                                     # policy type
                env,                                                                    # environment
                verbose=0,                                                              # without printing training details to the TUI
                tensorboard_log=tb_log,                                                 # tensorboard log file
                learning_rate=self.rl_alg_hyp['alpha'],                                 # learning rate
                gamma=self.rl_alg_hyp['gamma'],                                         # discount factor
                buffer_size=int(self.rl_alg_hyp['buffer_size']),                        # replay buffer size
                batch_size=int(self.rl_alg_hyp['batch_size']),                          # batch size
                learning_starts=int(self.rl_alg_hyp['learning_starts']),                # No. of steps before starting the training of actor/critic networks
                tau=self.rl_alg_hyp['tau'],                                             # soft update parameter
                train_freq=self.rl_alg_hyp['train_freq'],                               # training frequency
                ent_coef=self.rl_alg_hyp['ent_coeff'],                                  # entropy coefficient
                policy_kwargs=policy_kwargs,                                            # contains network hyperparameters
                use_sde=gSDE,                                                           # gSDE exploration  (0: False, 1: True)
                device=self.device,                                                     # CPU or GPU
                target_update_interval=self.rl_alg_hyp['train_freq'],                   # target network update interval
                seed=self.r_seed_train,                                                 # random seed
            )
        
        elif self.rl_alg == 'TQC':                      # import algorithm
            from sb3_contrib import TQC
            policy_kwargs = dict(activation_fn=activation_fn, net_arch=net_arch,
                            n_critics=int(self.rl_alg_hyp['n_critics']), n_quantiles=int(self.rl_alg_hyp['n_quantiles']))
            if self.rl_alg_hyp['gSDE'] == 0:
                gSDE = False
            else:
                gSDE = True
            model = TQC(
                "MultiInputPolicy",                                                         # policy type
                env,                                                                        # environment
                verbose=0,                                                                  # without printing training details to the TUI
                tensorboard_log=tb_log,                                                     # tensorboard log file
                top_quantiles_to_drop_per_net=int(self.rl_alg_hyp['top_quantiles_drop']),   # No. of top quantiles to drop
                learning_rate=self.rl_alg_hyp['alpha'],                                     # learning rate
                gamma=self.rl_alg_hyp['gamma'],                                             # discount factor
                buffer_size=int(self.rl_alg_hyp['buffer_size']),                            # replay buffer size
                batch_size=int(self.rl_alg_hyp['batch_size']),                              # batch size
                learning_starts=int(self.rl_alg_hyp['learning_starts']),                    # No. of steps before starting the training of actor/critic networks
                tau=self.rl_alg_hyp['tau'],                                                 # soft update parameter
                train_freq=self.rl_alg_hyp['train_freq'],                                   # training frequency
                ent_coef=self.rl_alg_hyp['ent_coeff'],                                      # entropy coefficient
                policy_kwargs=policy_kwargs,                                                # contain network hyperparameters
                use_sde=gSDE,                                                               # gSDE exploration  (0: False, 1: True)
                device=self.device,                                                         # CPU or GPU
                target_update_interval=self.rl_alg_hyp['train_freq'],                       # target network update interval
                seed=self.r_seed_train,                                                     # random seed
            )
        else:
            assert False, 'Algorithm is not implemented!'
        
        return model
    
    def load_model(self, env, tb_log):
        """
            Load pretrained Stable-Baselines3 model for RL training
            :param env: environment
            :param tb_log: # tensorboard log file
            :return model: Stable-Baselines3 model for RL training
        """

        ENV_PARAMS = EnvParams()

        # Load pretrained RL algorithms
        if self.rl_alg == 'DQN':
            from stable_baselines3 import DQN           # import algorithm
            model = DQN.load(self.path_files + self.str_inv_load, tensorboard_log=tb_log)
            print("Check replay buffer:")
            print(f"The loaded_model has {model.replay_buffer.size()} transitions in its buffer - before loading the replay buffer")
            model.load_replay_buffer(self.path_files + self.str_inv_load)
            print(f"The loaded_model has {model.replay_buffer.size()} transitions in its buffer - after loading the replay buffer")
            model.set_env(env)

        elif self.rl_alg == 'A2C':                      # import algorithm
            from stable_baselines3 import A2C
            model = A2C.load(self.path_files + self.str_inv_load, tensorboard_log=tb_log)
            model.set_env(env)

        elif self.rl_alg == 'PPO':                      # import algorithm
            from stable_baselines3 import PPO
            model = PPO.load(self.path_files + self.str_inv_load, tensorboard_log=tb_log)
            model.set_env(env)

        elif self.rl_alg == 'TD3':                      # import algorithm
            from stable_baselines3 import TD3
            model = TD3.load(self.path_files + self.str_inv_load, tensorboard_log=tb_log)
            print("Check replay buffer:")
            print(f"The loaded_model has {model.replay_buffer.size()} transitions in its buffer - before loading the replay buffer")
            model.load_replay_buffer(self.path_files + self.str_inv_load)
            print(f"The loaded_model has {model.replay_buffer.size()} transitions in its buffer - after loading the replay buffer")
            model.set_env(env)
        
        elif self.rl_alg == 'SAC':                      # import algorithm
            from stable_baselines3 import SAC
            model = SAC.load(self.path_files + self.str_inv_load, tensorboard_log=tb_log)
            print("Check replay buffer:")
            print(f"The loaded_model has {model.replay_buffer.size()} transitions in its buffer - before loading the replay buffer")
            model.load_replay_buffer(self.path_files + self.str_inv_load)
            print(f"The loaded_model has {model.replay_buffer.size()} transitions in its buffer - after loading the replay buffer")
            model.set_env(env)

        elif self.rl_alg == 'TQC':                      # import algorithm
            from sb3_contrib import TQC
            model = TQC.load(self.path_files + self.str_inv_load, tensorboard_log=tb_log)
            print("Check replay buffer:")
            print(f"The loaded_model has {model.replay_buffer.size()} transitions in its buffer - before loading the replay buffer")
            model.load_replay_buffer(self.path_files + self.str_inv_load)
            print(f"The loaded_model has {model.replay_buffer.size()} transitions in its buffer - after loading the replay buffer")
            model.set_env(env)
        else:
            assert False, 'Algorithm is not implemented!'

        return model
    
    def save_model(self, model):
        """
            Load pretrained Stable-Baselines3 model for RL training
            :param env: environment
            :param tb_log: # tensorboard log file
            :return model: Stable-Baselines3 model for RL training
        """

        print("Save final Model...")
        model.save(self.path_files + self.str_inv)
        if 'buffer_size' in self.rl_alg_hyp.keys():
            model.save_replay_buffer(self.path_files + self.str_inv)


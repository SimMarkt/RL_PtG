# ----------------------------------------------------------------------------------------------------------------
# RL_PtG: Deep Reinforcement Learning for Power-to-Gas dispatch optimization
# https://github.com/SimMarkt/RL_PtG

# rl_config_agent: 
# > Contains hyperparameters of the RL agents
# > Converts the config_agent.yaml data into a class object for further processing
# > Contains three additional functions to specify, load, and save the Stable-Baselines3 models
# ----------------------------------------------------------------------------------------------------------------

import numpy as np
import torch as th
import yaml

class AgentConfig:
    def __init__(self):
        # Load the environment configuration
        with open("config/config_agent.yaml", "r") as env_file:
            agent_config = yaml.safe_load(env_file)

        self.n_envs = agent_config['n_envs']                                # Number environments/workers for training
        assert agent_config['rl_alg'] in agent_config['hyperparameters'], f'Wrong algorithm specified - data/config_agent.yaml -> 
                                                                            model_conf : {agent_config['rl_alg']} must match {agent_config['hyperparameters'].keys()}'
        self.rl_alg = agent_config['rl_alg']                                # selected RL algorithm - already implemented [DQN, A2C, PPO, TD3, SAC, TQC]
        self.rl_alg_hyp = agent_config['hyperparameters'][self.rl_alg]      # hyperparameters of the algorithm
        

    def set_model(self, env, tb_log):
        """
            Specify the Stable-Baselines3 model for RL training
            :param env: environment
            :param tb_log: Tensorboard log file
            :seed_train: random seed for training
            :return model: Stable-Baselines3 model for RL training
        """
        
        # Implement the neural network architecture:
        #   Custom actor (pi) and value function (vf) networks with the
        #   same architecture net_arch and activation function (th.nn.ReLU, th.nn.Tanh)
        #   Note: an extra linear layer will be added on top of the pi and vf nets
        if self.rl_alg_hyp.activation == 'ReLU':
            activation_fn = th.nn.ReLU
        elif self.rl_alg_hyp.activation == 'Tanh':
            activation_fn = th.nn.Tanh
        else:
            assert False, "Type of activation function ({self.rl_alg_hyp.activation}) needs to be 'ReLU' or 'Tanh'! -> Check RL_PtG/config/config_agent.yaml"
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
            assert self.rl_alg_hyp['normalize_advantage'] in [False,True], f"Normalize advantage ({self.rl_alg_hyp['normalize_advantage']}) should be 'False' or 'True'! - Check RL_PtG/config/config_agent.yaml"
            assert self.rl_alg_hyp['gSDE'] in [False,True], f"gSDE exploration ({self.rl_alg_hyp['gSDE']}) should be 'False' or 'True'! - Check RL_PtG/config/config_agent.yaml"
            model = A2C(
                "MultiInputPolicy",                                                     # policy type
                env,                                                                    # environment
                verbose=0,                                                              # without printing training details to the TUI
                tensorboard_log=tb_log,                                                 # tensorboard log file
                learning_rate=self.rl_alg_hyp['alpha'],                                 # learning rate
                gamma=self.rl_alg_hyp['gamma'],                                         # discount factor
                n_steps=self.rl_alg_hyp['n_steps'],                                     # No. of steps of the n-step TD update
                gae_lambda=self.rl_alg_hyp['gae_lambda'],                               # factor for generalized advantage estimation
                normalize_advantage=self.rl_alg_hyp['normalize_advantage'],             # normalize advantage
                ent_coef=self.rl_alg_hyp['ent_coeff'],                                  # entropy coefficient
                policy_kwargs=policy_kwargs,                                            # contains network hyperparameters
                use_sde=self.rl_alg_hyp['gSDE'],                                        # gSDE exploration
                device=self.device,                                                     # CPU or GPU
                seed=self.r_seed_train,                                                 # random seed
            )

        elif self.rl_alg == 'PPO':                      # import algorithm
            from stable_baselines3 import PPO
            policy_kwargs = dict(activation_fn=activation_fn, net_arch=net_arch)
            n_steps = int(self.rl_alg_hyp.n_steps_f * self.rl_alg_hyp.batch_size)       # number of steps of the n-step TD update
            assert self.rl_alg_hyp['normalize_advantage'] in [False,True], f"Normalize advantage ({self.rl_alg_hyp['normalize_advantage']}) should be 'False' or 'True'! - Check RL_PtG/config/config_agent.yaml"
            assert self.rl_alg_hyp['gSDE'] in [False,True], f"gSDE exploration ({self.rl_alg_hyp['gSDE']}) should be 'False' or 'True'! - Check RL_PtG/config/config_agent.yaml"
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
                normalize_advantage=self.rl_alg_hyp['normalize_advantage'],             # normalize advantage
                ent_coef=self.rl_alg_hyp['ent_coeff'],                                  # entropy coefficient
                policy_kwargs=policy_kwargs,                                            # contains network hyperparameters
                use_sde=self.rl_alg_hyp['gSDE'],                                        # gSDE exploration
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
            assert self.rl_alg_hyp['gSDE'] in [False,True], f"gSDE exploration ({self.rl_alg_hyp['gSDE']}) should be 'False' or 'True'! - Check RL_PtG/config/config_agent.yaml"
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
                use_sde=self.rl_alg_hyp['gSDE'],                                        # gSDE exploration
                device=self.device,                                                     # CPU or GPU
                target_update_interval=self.rl_alg_hyp['train_freq'],                   # target network update interval
                seed=self.r_seed_train,                                                 # random seed
            )
        
        elif self.rl_alg == 'TQC':                      # import algorithm
            from sb3_contrib import TQC
            policy_kwargs = dict(activation_fn=activation_fn, net_arch=net_arch,
                            n_critics=int(self.rl_alg_hyp['n_critics']), n_quantiles=int(self.rl_alg_hyp['n_quantiles']))
            assert self.rl_alg_hyp['gSDE'] in [False,True], f"gSDE exploration ({self.rl_alg_hyp['gSDE']}) should be 'False' or 'True'! - Check RL_PtG/config/config_agent.yaml"
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
                use_sde=self.rl_alg_hyp['gSDE'],                                            # gSDE exploration
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
            :param tb_log: Tensorboard log file
            :return model: Stable-Baselines3 model for RL training
        """

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

    def get_hyper(self):
        """
            Gather and print algorithm hyperparameters and returns the values in a string for file identification
            :return str_set: hyperparameter settings in a string for file identification
        """

        # Set RL algorithms and specify hyperparameters
        print(f"    > Deep RL agorithm : {self.rl_alg}")
        print(f"        Learning rate : {self.rl_alg_hyp['alpha']}")
        print(f"        Discount factor : {self.rl_alg_hyp['gamma']}")
        if self.rl_alg == 'DQN': 
            print(f"        Initial exploration coefficient : {self.rl_alg_hyp['eps_init']}")
            print(f"        Final exploration coefficient : {self.rl_alg_hyp['eps_fin']}")
            print(f"        Ratio of the training set for exploration annealing : {self.rl_alg_hyp['eps_fra']}")
        if self.rl_alg in ['A2C','PPO','SAC','TQC']: print(f"        Entropy coefficient : {self.rl_alg_hyp['ent_coeff']}")
        if self.rl_alg in ['TD3']: print(f"        Exploration noise in the target   : {self.rl_alg_hyp['sigma_exp']}")
        if self.rl_alg in ['A2C']: print(f"        No. of steps of the n-step TD update : {int(self.rl_alg_hyp['n_steps'])}")
        if self.rl_alg in ['PPO']: print(f"        No. of steps of the n-step TD update : {int(self.rl_alg_hyp.n_steps_f * self.rl_alg_hyp.batch_size)} | n_steps_f = {int(self.rl_alg_hyp.n_steps_f)}")
        if self.rl_alg in ['DQN','TD3','SAC','TQC']: print(f"        Replay buffer size : {int(self.rl_alg_hyp['buffer_size'])}")
        if self.rl_alg in ['DQN','PPO','TD3','SAC','TQC']: print(f"        Batch size : {int(self.rl_alg_hyp['batch_size'])}")
        print(f"        No. of hidden layers : {int(self.rl_alg_hyp['hidden_layers'])}")
        print(f"        No. of hidden units : {int(self.rl_alg_hyp['hidden_units'])}")
        print(f"        Activation function : {int(self.rl_alg_hyp['activation'])}")
        if self.rl_alg in ['A2C','PPO']: print(f"        Factor for generalized advantage estimation : {self.rl_alg_hyp['gae_lambda']}")
        if self.rl_alg == 'PPO': print(f"        No. of epochs for mini batch training : {self.rl_alg_hyp['n_epoch']}")
        if self.rl_alg in ['A2C','PPO']: print(f"        Normalize advantage : {self.rl_alg_hyp['normalize_advantage']}")
        if self.rl_alg == 'TQC':
            print(f"        No. of quantiles : {int(self.rl_alg_hyp['n_quantiles'])}")
            print(f"        Top quantiles to drop : {int(self.rl_alg_hyp['top_quantiles_drop'])}")
            print(f"        No. of critics : {int(self.rl_alg_hyp['n_critics'])}")
        if self.rl_alg in ['DQN','TD3','SAC','TQC']: print(f"        Soft update coefficient : {self.rl_alg_hyp['tau']}")
        if self.rl_alg in ['DQN','TD3','SAC','TQC']: print(f"        Learning starts : {int(self.rl_alg_hyp['learning_starts'])}")
        if self.rl_alg in ['DQN','TD3','SAC','TQC']: print(f"        Training frequency : {int(self.rl_alg_hyp['train_freq'])}")
        if self.rl_alg == 'DQN': print(f"        Target network update interval : {int(self.rl_alg_hyp['target_update_interval'])}")
        if self.rl_alg in ['A2C','PPO','TD3','SAC','TQC']: print(f"        gSDE exploration  (0: False, 1: True) : {int(self.rl_alg_hyp['gSDE'])}")



def get_param_str(eps_sim_steps_train, seed):                           ################################------------------------------------------------------------
    """
    Returns the hyperparameter setting as a long string
    :return: hyperparameter setting
    """
    AgentConfig = AgentConfig()

    str_params_short = "_ep" + str(AgentConfig.eps_len_d) + \
                       "_al" + str(np.round(AgentConfig.alpha, 6)) + \
                       "_ga" + str(np.round(AgentConfig.gamma, 4)) + \
                       "_bt" + str(AgentConfig.batch_size) + \
                       "_bf" + str(AgentConfig.buffer_size) + \
                       "_et" + str(np.round(AgentConfig.ent_coeff, 5)) + \
                       "_hu" + str(AgentConfig.hidden_units) + \
                       "_hl" + str(AgentConfig.hidden_layers) + \
                       "_st" + str(AgentConfig.sim_step) + \
                       "_ac" + str(AgentConfig.activation) + \
                       "_ls" + str(AgentConfig.learning_starts) + \
                       "_tf" + str(AgentConfig.train_freq) + \
                       "_tau" + str(AgentConfig.tau) + \
                       "_cr" + str(AgentConfig.n_critics) + \
                       "_qu" + str(AgentConfig.n_quantiles) + \
                       "_qd" + str(AgentConfig.top_quantiles_drop) + \
                       "_gsd" + str(AgentConfig.gSDE) + \
                       "_sd" + str(seed)

    str_params_long = "\n     episode length=" + str(AgentConfig.eps_len_d) + \
                      "\n     alpha=" + str(AgentConfig.alpha) + \
                      "\n     gamma=" + str(AgentConfig.gamma) + \
                      "\n     batchsize=" + str(AgentConfig.batch_size) + \
                      "\n     replaybuffer=" + str(AgentConfig.buffer_size) + \
                      " (#ofEpisodes=" + str(AgentConfig.buffer_size / eps_sim_steps_train) + ")" + \
                      "\n     coeff_ent=" + str(AgentConfig.ent_coeff) + \
                      "\n     hiddenunits=" + str(AgentConfig.hidden_units) + \
                      "\n     hiddenlayers=" + str(AgentConfig.hidden_layers) + \
                      "\n     sim_step=" + str(AgentConfig.sim_step) + \
                      "\n     activation=" + str(AgentConfig.activation) + " (0=Relu, 1=tanh" + ")" + \
                      "\n     learningstarts=" + str(AgentConfig.learning_starts) + \
                      "\n     training_freq=" + str(AgentConfig.train_freq) + \
                      "\n     tau=" + str(AgentConfig.tau) + \
                      "\n     n_critics=" + str(AgentConfig.n_critics) + \
                      "\n     n_quantiles=" + str(AgentConfig.n_quantiles) + \
                      "\n     top_quantiles_to_drop_per_net=" + str(AgentConfig.top_quantiles_drop) + \
                      "\n     g_SDE=" + str(AgentConfig.gSDE) + \
                      "\n     seed=" + str(seed)

    return str_params_short, str_params_long


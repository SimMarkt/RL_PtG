# ----------------------------------------------------------------------------------------------------------------
# RL_PtG: Deep Reinforcement Learning for Power-to-Gas dispatch optimization
# https://github.com/SimMarkt/RL_PtG

# rl_hyp_param: 
# > Contains hyperparameters of the RL agents
# > Converts the config_agent.yaml data into a class object for further processing
# ----------------------------------------------------------------------------------------------------------------


import numpy as np
import yaml


class HypParams:
    def __init__(self):
        # Load the agent configuration
        with open("config/config_agent.yaml", "r") as env_file:
            agent_config = yaml.safe_load(env_file)
        
        self.str_inv = agent_config['str_inv']          # string for denoting your training results and models to a specific investigation #################---------------
        self.model_conf = agent_config['model_conf']    # training setup: "simple_train", "save_model", "load_model", "save_load_model"

        self.n_envs = agent_config['n_envs']                    # number environments/workers for training
        self.rl_alg = agent_config['rl_alg']     # selected RL algorithm - already implemented [DQN, A2C, PPO, TD3, SAC, TQC]
        if self.rl_alg != agent_config['hyperparameters']:### überprüfe auf hinterlegte Algorithmen #####--------
            assert False, f'Wrong algorithm specified - data/config_agent.yaml > rl_alg : {agent_config['hyperparameters']} must match ????' #### leite liste mit allen Algorithmen aus dict ab #####--------
        
        
state_change_penalty : 0.0   # Factor which enables reward penalty during training (if state_change_penalty = 0.0: No reward penalty; if state_change_penalty > 0.0: Reward penalty on mode transitions;)
r_seed_train : [3654, 467, 9327, 5797, 249, 9419]         # random seeds for neural network initialization (and environment randomness) of the training set
r_seed_test : [605, 5534, 2910, 7653, 8936, 1925]         # random seeds for neural network initialization (and environment randomness) of the validation and test sets

        self.sim_step = agent_config['sim_step']                # Frequency for taking an action in [s]
        self.eps_len_d = agent_config['eps_len_d']              # No. of days in an episode (episodes are randomly selected from the entire training data set without replacement)
        self.state_change_penalty = agent_config['state_change_penalty']  


        # hyperparameters of the algorithm
        self.alpha = 0.00044
        self.gamma = 0.9639
        self.ent_coeff = 0.00047
        self.buffer_size = 6165170
        self.batch_size = 290
        self.hidden_layers = 3
        self.hidden_units = 708
        self.activation = 0
        self.top_quantiles_drop = 2
        self.n_quantiles = 30
        self.train_freq = 1
        self.n_critics = 2             # "n_critics"              # Basecase: 2
        self.tau = 0.005            # "n_quantiles"              # Basecase: 25
        self.learning_starts = 100      # "top_quantiles_to_drop_per_net"              # Basecase: 2
        self.gSDE = 0                   # gSDE exploration  (0: False, 1: True)

        # reward penalty during training
        self.state_change_penalty = 0.0

        # random seeds
        self.r_seed_train = [3654, 467, 9327, 5797, 249, 9419]         # for training
                                        # [3654, 467, 9327, 5797, 249, 9419, 676, 1322, 9010, 4021]
        self.r_seed_test = [605, 5534, 2910, 7653, 8936, 1925]         # for evaluation
                                        # [605, 5534, 2910, 7653, 8936, 1925, 4286, 7435, 6276, 3008, 361]



n_envs : 6                    # Number environments/workers for training
rl_alg : TQC    # selected RL algorithm - already implemented [DQN, A2C, PPO, TD3, SAC, TQC]
sim_step : 600               # Frequency for taking an action in [s]
eps_len_d : 32               # No. of days in an episode (episodes are randomly selected from the entire training data set without replacement)
state_change_penalty : 0.0   # Factor which enables reward penalty during training (if state_change_penalty = 0.0: No reward penalty; if state_change_penalty > 0.0: Reward penalty on mode transitions;)
r_seed_train : [3654, 467, 9327, 5797, 249, 9419]         # random seeds for neural network initialization (and environment randomness) of the training set
r_seed_test : [605, 5534, 2910, 7653, 8936, 1925]         # random seeds for neural network initialization (and environment randomness) of the validation and test sets

hyperparameters:    # Implemented hyperparameters of the RL algorithms used
    DQN :
        action_type : discrete        # the type of action: "discrete" or "continuous"
        alpha : 0.0002                  # learning rate
        gamma : 0.9728                  # discount factor
        buffer_size : 1000000           # replay buffer size
        batch_size : 544                # batch size
        eps_init : 0.1817               # initial exploration coefficient
        eps_fin : 0.00268               # final exploration coefficient
        eps_fra : 0.00434               # ratio of the training set for exploration annealing
        hidden_layers : 7               # No. of hidden layers
        hidden_units : 366              # No. of hidden units
        activation : 0                  # activation function (0: ReLU, 1: Tanh)
        train_freq : 5                  # training frequency
        target_update_interval : 113204 # target network update interval
        tau : 1.0                       # soft update parameter
        learning_starts : 50000         # No. of steps before starting the training of actor/critic networks
    A2C :
        action_type : discrete        # the type of action: "discrete" or "continuous"
        alpha : 0.00021                 # learning rate
        gamma : 0.9393                  # discount factor
        ent_coeff : 0.00004             # entropy coefficient
        n_steps : 658                   # No. of steps of the n-step TD update
        hidden_layers : 4               # No. of hidden layers
        hidden_units : 808              # No. of hidden units
        activation : 1                  # activation function (0: ReLU, 1: Tanh)
        gae_lambda : 0.9819             # factor for generalized advantage estimation
        normalize_advantage : 1         # normalize advantage (0: Off, 1: On)
        gSDE : 0                        # gSDE exploration  (0: False, 1: True)
    PPO : 
        action_type : discrete        # the type of action: "discrete" or "continuous"
        alpha : 0.00005                 # learning rate
        gamma : 0.973                   # discount factor
        ent_coeff : 0.00001             # entropy coefficient 
        n_steps_f : 21                  # factor for determining the No. of steps of the n-step TD update (n_steps = n_steps_f * batch_size)
        batch_size : 203                # batch size
        hidden_layers : 2               # No. of hidden layers
        hidden_units : 358              # No. of hidden units
        activation : 0                  # activation function (0: ReLU, 1: Tanh)
        gae_lambda : 0.8002             # factor for generalized advantage estimation
        n_epoch : 13                    # No. of epochs for mini batch training
        normalize_advantage : 0         # normalize advantage (0: Off, 1: On)
        gSDE : 0                        # gSDE exploration  (0: False, 1: True)
    TD3 :
        action_type : continuous      # the type of action: "discrete" or "continuous"
        alpha : 0.00015                 # learning rate
        gamma : 0.9595                  # discount factor
        buffer_size : 19984440          # replay buffer size
        batch_size : 257                # batch size
        hidden_layers : 3               # No. of hidden layers
        hidden_units : 743              # No. of hidden units
        activation : 0                  # activation function (0: ReLU, 1: Tanh)
        train_freq : 9                  # training frequency
        tau : 0.005                     # soft update parameter
        learning_starts : 100           # No. of steps before starting the training of actor/critic networks
        sigma_exp : 0.19                # exploration noise in the target          
    SAC :
        action_type : continuous      # the type of action: "discrete" or "continuous"
        alpha : 0.0001                  # learning rate
        gamma : 0.9628                  # discount factor
        ent_coeff : 2                   # entropy coefficient (choose either values within [0,1] or the automatic adjustment using [2])
        buffer_size : 2267734           # replay buffer size
        batch_size : 470                # batch size
        hidden_layers : 6               # No. of hidden layers
        hidden_units : 233              # No. of hidden units
        activation : 1                  # activation function (0: ReLU, 1: Tanh)
        train_freq : 1                  # training frequency
        tau : 0.005                     # soft update parameter
        learning_starts : 100           # No. of steps before starting the training of actor/critic networks
        gSDE : 0                        # gSDE exploration  (0: False, 1: True)

    TQC :
        action_type : continuous      # the type of action: "discrete" or "continuous"
        alpha : 0.00044                 # learning rate
        gamma : 0.9639                  # discount factor
        ent_coeff : 0.00047             # entropy coefficient  (choose either values within [0,1] or the automatic adjustment using [2])
        buffer_size : 6165170           # replay buffer size
        batch_size : 290                # batch size
        hidden_layers : 3               # No. of hidden layers
        hidden_units : 708              # No. of hidden units
        activation : 0                  # activation function (0: ReLU, 1: Tanh)
        top_quantiles_drop : 2          # No. of top quantiles to drop
        n_quantiles : 30                # No. of quantiles for distributed value estimation
        train_freq : 1                  # training frequency
        n_critics : 2                   # No. of critics
        tau : 0.005                     # soft update parameter
        learning_starts : 100           # No. of steps before starting the training of actor/critic networks
        gSDE : 0                        # gSDE exploration  (0: False, 1: True)


def get_param_str(eps_sim_steps_train, seed):
    """
    Returns the hyperparameter setting as a long string
    :return: hyperparameter setting
    """
    HYPER_PARAMS = HypParams()

    str_params_short = "_ep" + str(HYPER_PARAMS.eps_len_d) + \
                       "_al" + str(np.round(HYPER_PARAMS.alpha, 6)) + \
                       "_ga" + str(np.round(HYPER_PARAMS.gamma, 4)) + \
                       "_bt" + str(HYPER_PARAMS.batch_size) + \
                       "_bf" + str(HYPER_PARAMS.buffer_size) + \
                       "_et" + str(np.round(HYPER_PARAMS.ent_coeff, 5)) + \
                       "_hu" + str(HYPER_PARAMS.hidden_units) + \
                       "_hl" + str(HYPER_PARAMS.hidden_layers) + \
                       "_st" + str(HYPER_PARAMS.sim_step) + \
                       "_ac" + str(HYPER_PARAMS.activation) + \
                       "_ls" + str(HYPER_PARAMS.learning_starts) + \
                       "_tf" + str(HYPER_PARAMS.train_freq) + \
                       "_tau" + str(HYPER_PARAMS.tau) + \
                       "_cr" + str(HYPER_PARAMS.n_critics) + \
                       "_qu" + str(HYPER_PARAMS.n_quantiles) + \
                       "_qd" + str(HYPER_PARAMS.top_quantiles_drop) + \
                       "_gsd" + str(HYPER_PARAMS.gSDE) + \
                       "_sd" + str(seed)

    str_params_long = "\n     episode length=" + str(HYPER_PARAMS.eps_len_d) + \
                      "\n     alpha=" + str(HYPER_PARAMS.alpha) + \
                      "\n     gamma=" + str(HYPER_PARAMS.gamma) + \
                      "\n     batchsize=" + str(HYPER_PARAMS.batch_size) + \
                      "\n     replaybuffer=" + str(HYPER_PARAMS.buffer_size) + \
                      " (#ofEpisodes=" + str(HYPER_PARAMS.buffer_size / eps_sim_steps_train) + ")" + \
                      "\n     coeff_ent=" + str(HYPER_PARAMS.ent_coeff) + \
                      "\n     hiddenunits=" + str(HYPER_PARAMS.hidden_units) + \
                      "\n     hiddenlayers=" + str(HYPER_PARAMS.hidden_layers) + \
                      "\n     sim_step=" + str(HYPER_PARAMS.sim_step) + \
                      "\n     activation=" + str(HYPER_PARAMS.activation) + " (0=Relu, 1=tanh" + ")" + \
                      "\n     learningstarts=" + str(HYPER_PARAMS.learning_starts) + \
                      "\n     training_freq=" + str(HYPER_PARAMS.train_freq) + \
                      "\n     tau=" + str(HYPER_PARAMS.tau) + \
                      "\n     n_critics=" + str(HYPER_PARAMS.n_critics) + \
                      "\n     n_quantiles=" + str(HYPER_PARAMS.n_quantiles) + \
                      "\n     top_quantiles_to_drop_per_net=" + str(HYPER_PARAMS.top_quantiles_drop) + \
                      "\n     g_SDE=" + str(HYPER_PARAMS.gSDE) + \
                      "\n     seed=" + str(seed)

    return str_params_short, str_params_long




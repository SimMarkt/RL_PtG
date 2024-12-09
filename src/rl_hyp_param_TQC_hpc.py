# Contains hyperparameters of RL optimization of Power-to-Gas using TQC
import numpy as np
import pandas as pd
import os


class HypParams:
    def __init__(self):
        self.str_vary_former = "TQC_hpo_"
        self.model_conf = "simple_train"  # "simple_train", "save_model", "load_model", "save_load_model"

        self.rl_alg = "TQC"

        # (hyper)parameters with regard to the agent-environment interaction
        self.sim_step = 600               # Frequency for taking an action in sec
        self.eps_len_d = 32               # No. of days in an episode

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



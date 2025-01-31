# ----------------------------------------------------------------------------------------------------------------
# RL_PtG: Deep Reinforcement Learning for Power-to-Gas dispatch optimization
# https://github.com/SimMarkt/RL_PtG

# rl_utils: 
# > Utiliy/helper functions
# ----------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import math

from src.rl_config_agent import AgentConfiguration
from src.rl_config_env import EnvConfiguration
from src.rl_config_train import TrainConfiguration
from src.rl_opt import calculate_optimum

def import_market_data(csvfile: str, type: str, path: str):
    """
        Import data of day-ahead prices for electricity
        :param csvfile: Name of the .csv files containing energy market data ["Time [s]"; <data>]
        :param type: market data type
        :param path: String with the RL_PtG project path
        :param TrainConfig: Training configuration in a class object
        :return arr: np.array with market data
    """

    file_path = path + "/" + csvfile
    df = pd.read_csv(file_path, delimiter=";", decimal=".")
    df["Time"] = pd.to_datetime(df["Time"], format="%d-%m-%Y %H:%M")

    if type == "elec": # electricity price data
        arr = df["Day-Ahead-price [Euro/MWh]"].values.astype(float) / 10        # Convert Euro/MWh into ct/kWh
    elif type == "gas": # gas price data
        arr = df["THE_DA_Gas [Euro/MWh]"].values.astype(float) / 10             # Convert Euro/MWh into ct/kWh
    elif type == "eua": # EUA price data
        arr = df["EUA_CO2 [Euro/t]"].values.astype(float)
    else:
        assert False, "Market data type not specified appropriately [elec, gas, eua]!"

    return arr


def import_data(csvfile: str, path: str):
    """
        Import experimental methanation data for state changes
        :param csvfile: name of the .csv files containing process data ["Time [s]", "T_cat [°C]", "n_h2 [mol/s]", "n_ch4 [mol/s]", "n_h2_res [mol/s]", "m_DE [kg/h]", "Pel [W]"]
        :param path: String with the RL_PtG project path
        :return arr: Numpy array with operational data
    """
    # Import historic Data from csv file
    file_path = path + "/" + csvfile
    df = pd.read_csv(file_path, delimiter=";", decimal=".")

    # Transform data to numpy array
    t = df["Time [s]"].values.astype(float)                         # Time
    t_cat = df["T_cat [gradC]"].values.astype(float)                # Maximum catalyst temperature in the methanation reactor system [°C]
    n_h2 = df["n_h2 [mol/s]"].values.astype(float)                  # Hydrogen reactant molar flow [mol/s]
    n_ch4 = df["n_ch4 [mol/s]"].values.astype(float)                # Methane product molar flow [mol/s]
    n_h2_res = df["n_h2_res [mol/s]"].values.astype(float)          # Hydrogen product molar flow (residues) [mol/s]    
    m_h2o = df["m_DE [kg/h]"].values.astype(float)                  # Water mass flow [kg/h]
    p_el = df["Pel [W]"].values.astype(float)                       # Electrical power consumption for heating the methanation plant [W]
    arr = np.c_[t, t_cat, n_h2, n_ch4, n_h2_res, m_h2o, p_el]

    return arr


def load_data(EnvConfig, TrainConfig):
    """
        Loads historical market data and experimental data of methanation operation
        :return dict_price_data: dictionary with market data
                dict_op_data: dictionary with data of dynamic methanation operation
    """

    print("---Load data")
    path = TrainConfig.path
    # Load historical market data for electricity, gas and EUA ##########################################MAKE SHORTER##################
    dict_price_data = {'el_price_train': import_market_data(EnvConfig.datafile_path_train_el, "elec", path),     # Electricity prices of the training set
                       'el_price_val': import_market_data(EnvConfig.datafile_path_val_el, "elec", path),           # Electricity prices of the validation set
                       'el_price_test': import_market_data(EnvConfig.datafile_path_test_el, "elec", path),       # Electricity prices of the test set
                       'gas_price_train': import_market_data(EnvConfig.datafile_path_train_gas, "gas", path),    # Gas prices of the training set
                       'gas_price_val': import_market_data(EnvConfig.datafile_path_val_gas, "gas", path),          # Gas prices of the validation set
                       'gas_price_test': import_market_data(EnvConfig.datafile_path_test_gas, "gas", path),      # Gas prices of the test set
                       'eua_price_train': import_market_data(EnvConfig.datafile_path_train_eua, "eua", path),    # EUA prices of the training set
                       'eua_price_val': import_market_data(EnvConfig.datafile_path_val_eua, "eua", path),          # EUA prices of the validation set
                       'eua_price_test': import_market_data(EnvConfig.datafile_path_test_eua, "eua", path)}      # EUA prices of the test set

    # Load experimental methanation data for state changes  ##########################################MAKE SHORTER##################
    dict_op_data = {'startup_cold': import_data(EnvConfig.datafile_path2, path),     # Cold start
                    'startup_hot': import_data(EnvConfig.datafile_path3, path),      # Hot start
                    'cooldown': import_data(EnvConfig.datafile_path4, path),         # Cooldown
                    'standby_down': import_data(EnvConfig.datafile_path5, path),     # Standby dataset for high temperatures to standby
                    'standby_up': import_data(EnvConfig.datafile_path6, path),       # Standby dataset for low temperatures to standby
                    'op1_start_p': import_data(EnvConfig.datafile_path7, path),      # Partial load - warming up
                    'op2_start_f': import_data(EnvConfig.datafile_path8, path),      # Full load - warming up
                    'op3_p_f': import_data(EnvConfig.datafile_path9, path),          # Load change: Partial -> Full
                    'op4_p_f_p_5': import_data(EnvConfig.datafile_path10, path),     # Load change: Partial -> Full -> Partial (Return after 5 min)
                    'op5_p_f_p_10': import_data(EnvConfig.datafile_path11, path),    # Load change: Partial -> Full -> Partial (Return after 10 min)
                    'op6_p_f_p_15': import_data(EnvConfig.datafile_path12, path),    # Load change: Partial -> Full -> Partial (Return after 15 min)
                    'op7_p_f_p_22': import_data(EnvConfig.datafile_path13, path),    # Load change: Partial -> Full -> Partial (Return after 22 min)
                    'op8_f_p': import_data(EnvConfig.datafile_path14, path),         # Load change: Full -> Partial
                    'op9_f_p_f_5': import_data(EnvConfig.datafile_path15, path),     # Load change: Full -> Partial -> Full (Return after 5 min)
                    'op10_f_p_f_10': import_data(EnvConfig.datafile_path16, path),   # Load change: Full -> Partial -> Full (Return after 10 min)
                    'op11_f_p_f_15': import_data(EnvConfig.datafile_path17, path),   # Load change: Full -> Partial -> Full (Return after 15 min)
                    'op12_f_p_f_20': import_data(EnvConfig.datafile_path18, path)}   # Load change: Full -> Partial -> Full (Return after 20 min)

    if EnvConfig.scenario == 2:  # Fixed gas prices    ##########################################MAKE SHORTER##################
        dict_price_data['gas_price_train'] = np.ones(len(dict_price_data['gas_price_train'])) * EnvConfig.ch4_price_fix
        dict_price_data['gas_price_val'] = np.ones(len(dict_price_data['gas_price_val'])) * EnvConfig.ch4_price_fix
        dict_price_data['gas_price_test'] = np.ones(len(dict_price_data['gas_price_test'])) * EnvConfig.ch4_price_fix
    elif EnvConfig.scenario == 3:  # Gas and EUA prices = 0
        dict_price_data['gas_price_train'] = np.zeros(len(dict_price_data['gas_price_train']))
        dict_price_data['gas_price_val'] = np.zeros(len(dict_price_data['gas_price_val']))
        dict_price_data['gas_price_test'] = np.zeros(len(dict_price_data['gas_price_test']))
        dict_price_data['eua_price_train'] = np.zeros(len(dict_price_data['eua_price_train']))
        dict_price_data['eua_price_val'] = np.zeros(len(dict_price_data['eua_price_test']))
        dict_price_data['eua_price_test'] = np.zeros(len(dict_price_data['eua_price_test']))

    # For Reward level calculation -> Sets height of the reward penalty
    dict_price_data['el_price_reward_level'] = EnvConfig.r_0_values['el_price']
    dict_price_data['gas_price_reward_level'] = EnvConfig.r_0_values['gas_price']
    dict_price_data['eua_price_reward_level'] = EnvConfig.r_0_values['eua_price']

    # Check if training set is divisible by the episode length
    min_train_len = 6           # Minimum No. of days in the training set 
    EnvConfig.train_len_d = len(dict_price_data['gas_price_train']) - min_train_len # Training uses min_train_len days less than the data size to always allow enough space for the price forecast of Day-ahead market data
    assert EnvConfig.train_len_d > 0, f'The training set size must be greater than {min_train_len} days'
    if EnvConfig.train_len_d % EnvConfig.eps_len_d != 0:
        # Find all possible divisors of EnvConfig.train_len_d
        divisors = [i for i in range(1, EnvConfig.train_len_d + 1) if EnvConfig.train_len_d % i == 0]
        assert False, f'The training set size {EnvConfig.train_len_d} must be divisible by the episode length - data/config_env.yaml -> eps_len_d : {EnvConfig.eps_len_d}; Possible divisors are: {divisors}'
    # Ensure that the training set is larger or at least equal to the defined episode length  
    assert EnvConfig.train_len_d >= EnvConfig.eps_len_d, f'Training set size ({EnvConfig.train_len_d}) must be larger or at least equal to the defined episode length ({EnvConfig.eps_len_d})!'

    return dict_price_data, dict_op_data

class Preprocessing():
    """
        A class that contains variables and functions for preprocessing of energy market and process data
    """
    def __init__(self, dict_price_data, dict_op_data, AgentConfig, EnvConfig, TrainConfig):
        """
            Initialization of variables
            :param dict_price_data: Dictionary with market data
            :param dict_op_data: Dictionary with dynamic process data
            :param AgentConfig: Agent configuration in a class object
            :param EnvConfig: Environment configuration in a class object
            :param TrainConfig: Training configuration in a class object
        """
        # Initialization
        self.AgentConfig = AgentConfig
        self.EnvConfig = EnvConfig
        self.TrainConfig = TrainConfig
        self.dict_price_data = dict_price_data
        self.dict_op_data = dict_op_data
        self.dict_pot_r_b = None                    # dictionary with potential reward [pot_rew...] and boolean reward identifier [part_full_b...]
        self.r_level = None                         # Sets the general height of the reward penalty according to electricity, (S)NG, and EUA price levels
        # e_r_b_train/e_r_b_val/e_r_b_test: (hourly values)
        #   np.array which stores elec. price data, potential reward, and boolean identifier
        #   Dimensions = [Type of data] x [No. of day-ahead values] x [historical values]
        #       Type of data = [el_price, pot_rew, part_full_b]
        #       No. of day-ahead values = EnvConfig.price_ahead
        #       historical values = No. of values in the electricity price data set
        self.e_r_b_train = None
        self.e_r_b_val = None
        self.e_r_b_test = None
        # g_e_train/g_e_val/g_e_test: (daily values)
        #   np.array which stores gas and EUA price data
        #   Dimensions = [Type of data] x [No. of day-ahead values] x [historical values]
        #       Type of data = [gas_price, pot_rew, part_full_b]
        #       No. of day-ahead values = 2 (today and tomorrow)
        #       historical values = No. of values in the gas/EUA price data set
        self.g_e_train = None
        self.g_e_val = None
        self.g_e_test = None
        # Variables for division of the entire training set into different, randomly picked subsets for episodic learing
        self.eps_sim_steps_train = None         # Number of steps in the training set per episode
        self.eps_sim_steps_val = None            # Number of steps in the validation set
        self.eps_sim_steps_test = None          # Number of steps in the test set
        self.eps_ind = None                     # Contains indexes of the randomly ordered training subsets
        self.overhead_factor = 3                # Overhead of self.eps_ind - To account for randomn selection of the different processes in multiprocessing (need to be an integer)
        assert isinstance(self.overhead_factor, int), f"Episode overhead ({self.overhead_factor}) must be an integer!"
        self.n_eps = None                       # Episode length in seconds
        self.num_loops = None                   # No. of loops over the total training set during training

        # For Multiprocessing: self.n_eps_loops allows for definition of different eps_ind for different processes (see RL_PtG\env\ptg_gym_env.py)
        self.n_eps_loops = None                 # Total No. of episodes over the entire training procedure 

        self.preprocessing_rew()                                           # Calculate potential reward and boolean identifier
        self.preprocessing_array()                                         # Transform Day-ahead datasets into np.arrays for calculation purposes
        self.define_episodes()                                             # define episodes and indices for choosing subsets of the training set randomly          
        

    def preprocessing_rew(self):
        """
            Data preprocessing including the computation of a potential reward, which signifies the maximum reward the
            Power-to-Gas plant can yield in either partial load [part_full_b... = 0] or full load [part_full_b... = 1]
            :return dict_pot_r_b: dictionary with potential reward [pot_rew...] and boolean reward identifier [part_full_b...]
            :return r_level: ###################################################################################################################################?????????
        """

        # Compute methanation operation data for theoretical optimum (ignoring dynamics) ##########################################MAKE SHORTER##################
        # calculate_optimum() had been excluded from Preprocessing() and placed in the different rl_opt.py file for the sake of clarity
        print("---Calculate the theoretical optimum, the potential reward, and the load identifier")
        stats_dict_opt_train = calculate_optimum(self.dict_price_data['el_price_train'], self.dict_price_data['gas_price_train'],
                                                self.dict_price_data['eua_price_train'], "Training")
        stats_dict_opt_val = calculate_optimum(self.dict_price_data['el_price_val'], self.dict_price_data['gas_price_val'],
                                                self.dict_price_data['eua_price_val'], "Validation")
        stats_dict_opt_test = calculate_optimum(self.dict_price_data['el_price_test'], self.dict_price_data['gas_price_test'],
                                                self.dict_price_data['eua_price_test'], "Test")
        stats_dict_opt_level = calculate_optimum(self.dict_price_data['el_price_reward_level'], self.dict_price_data['gas_price_reward_level'],
                                                self.dict_price_data['eua_price_reward_level'], "reward_Level")

        # Store data sets with future values of the potential reward on the two different load levels and
        # data sets of a boolean identifier of future values of the potential reward in a dictionary
        # Pseudo code for part_full_b_... calculation:
        #       if pot_rew_... <= 0:
        #           part_full_b_... = -1
        #       else:
        #           if (pot_rew_... in full load) < (pot_rew... in partial load):
        #               part_full_b_... = 0
        #           else:
        #               part_full_b_... = 1
        self.dict_pot_r_b = {                                                                       ##########################################MAKE SHORTER##################
            'pot_rew_train': stats_dict_opt_train['Meth_reward_stats'],
            'part_full_b_train': stats_dict_opt_train['partial_full_b'],
            'pot_rew_val': stats_dict_opt_val['Meth_reward_stats'],
            'part_full_b_val': stats_dict_opt_val['partial_full_b'],
            'pot_rew_test': stats_dict_opt_test['Meth_reward_stats'],
            'part_full_b_test': stats_dict_opt_test['partial_full_b'],
        }

        self.r_level = stats_dict_opt_level['Meth_reward_stats']
        # multiple_plots(stats_dict_opt_train, 3600, "Opt_Training_set_sen" + str(EnvConfig.scenario))         ################# Include Graphics as default ##############
        # multiple_plots(stats_dict_opt_test, 3600, "Opt_Test_set_sen" + str(EnvConfig.scenario))              ################# Include Graphics as default ##############


    def preprocessing_array(self):
        """
        Transforms dictionaries to np.arrays for computational purposes
        """

        # Multi-Dimensional Array (3D) which stores day-ahead electricity price data as well as day-ahead potential reward
        # and boolean identifier for the entire training and test set
        # e.g. e_r_b_train[0, 5, 156] represents the future value of the electricity price [0,-,-] in 4 hours [-,5,-] at the
        # 156ths entry of the electricity price data set             ##########################################MAKE SHORTER##################
        self.e_r_b_train = np.zeros((3, self.EnvConfig.price_ahead, self.dict_price_data['el_price_train'].shape[0] - self.EnvConfig.price_ahead))
        self.e_r_b_val = np.zeros((3, self.EnvConfig.price_ahead, self.dict_price_data['el_price_val'].shape[0] - self.EnvConfig.price_ahead))
        self.e_r_b_test = np.zeros((3, self.EnvConfig.price_ahead, self.dict_price_data['el_price_test'].shape[0] - self.EnvConfig.price_ahead))

        for i in range(self.EnvConfig.price_ahead):     ##########################################MAKE SHORTER##################
            self.e_r_b_train[0, i, :] = self.dict_price_data['el_price_train'][i:(-self.EnvConfig.price_ahead + i)]
            self.e_r_b_train[1, i, :] = self.dict_pot_r_b['pot_rew_train'][i:(-self.EnvConfig.price_ahead + i)]
            self.e_r_b_train[2, i, :] = self.dict_pot_r_b['part_full_b_train'][i:(-self.EnvConfig.price_ahead + i)]
            self.e_r_b_val[0, i, :] = self.dict_price_data['el_price_val'][i:(-self.EnvConfig.price_ahead + i)]
            self.e_r_b_val[1, i, :] = self.dict_pot_r_b['pot_rew_val'][i:(-self.EnvConfig.price_ahead + i)]
            self.e_r_b_val[2, i, :] = self.dict_pot_r_b['part_full_b_val'][i:(-self.EnvConfig.price_ahead + i)]
            self.e_r_b_test[0, i, :] = self.dict_price_data['el_price_test'][i:(-self.EnvConfig.price_ahead + i)]
            self.e_r_b_test[1, i, :] = self.dict_pot_r_b['pot_rew_test'][i:(-self.EnvConfig.price_ahead + i)]
            self.e_r_b_test[2, i, :] = self.dict_pot_r_b['part_full_b_test'][i:(-self.EnvConfig.price_ahead + i)]

        # Multi-Dimensional Array (3D) which stores day-ahead gas and eua price data for the entire training and test set        ##########################################MAKE SHORTER##################
        self.g_e_train = np.zeros((2, 2, self.dict_price_data['gas_price_train'].shape[0] - 1))
        self.g_e_val = np.zeros((2, 2, self.dict_price_data['gas_price_val'].shape[0] - 1))
        self.g_e_test = np.zeros((2, 2, self.dict_price_data['gas_price_test'].shape[0] - 1))

        self.g_e_train[0, 0, :] = self.dict_price_data['gas_price_train'][:-1]     ##########################################MAKE SHORTER##################
        self.g_e_train[1, 0, :] = self.dict_price_data['eua_price_train'][:-1]
        self.g_e_val[0, 0, :] = self.dict_price_data['gas_price_val'][:-1]
        self.g_e_val[1, 0, :] = self.dict_price_data['eua_price_val'][:-1]
        self.g_e_test[0, 0, :] = self.dict_price_data['gas_price_test'][:-1]
        self.g_e_test[1, 0, :] = self.dict_price_data['eua_price_test'][:-1]
        self.g_e_train[0, 1, :] = self.dict_price_data['gas_price_train'][1:]
        self.g_e_train[1, 1, :] = self.dict_price_data['eua_price_train'][1:]
        self.g_e_val[0, 1, :] = self.dict_price_data['gas_price_val'][1:]
        self.g_e_val[1, 1, :] = self.dict_price_data['eua_price_val'][1:]
        self.g_e_test[0, 1, :] = self.dict_price_data['gas_price_test'][1:]
        self.g_e_test[1, 1, :] = self.dict_price_data['eua_price_test'][1:]


    def define_episodes(self):
        """
            Defines settings for training and evaluation episodes
        """

        print("---Define episodes and step size limits")
        # No. of days in the test set ("-1" excludes the day-ahead overhead)
        val_len_d = len(self.dict_price_data['gas_price_val']) - 1
        test_len_d = len(self.dict_price_data['gas_price_test']) - 1

        # Split up the entire training set into several smaller subsets which represents an own episodes
        self.n_eps = int(self.EnvConfig.train_len_d / self.EnvConfig.eps_len_d)  # Number of training subsets/episodes per training procedure
        
        self.eps_len = 24 * 3600 * self.EnvConfig.eps_len_d  # episode length in seconds

        # Number of steps in train and test sets per episode
        self.eps_sim_steps_train = int(self.eps_len / self.EnvConfig.sim_step)
        self.eps_sim_steps_val = int(24 * 3600 * val_len_d / self.EnvConfig.sim_step)
        self.eps_sim_steps_test = int(24 * 3600 * test_len_d /self. EnvConfig.sim_step)

        # Define total number of steps for all workers together
        self.num_loops = self.TrainConfig.train_steps / (self.eps_sim_steps_train * self.n_eps)      # Number of loops over the total training set
        print("    > Total number of training steps =", self.TrainConfig.train_steps)
        print("    > No. of loops over the entire training set =", round(self.num_loops,3))
        print("    > Training steps per episode =", self.eps_sim_steps_train)
        print("    > Steps in the evaluation set =", self.eps_sim_steps_test, "\n")

        # Create random selection routine with replacement for the different training subsets
        self.rand_eps_ind()

        # For Multiprocessing, eps_ind should not shared between different processes
        self.n_eps_loops = self.n_eps * int(self.num_loops)  # Allows for definition of different eps_ind in Multiprocessing (see RL_PtG\env\ptg_gym_env.py)


    def rand_eps_ind(self):
        """
        The agent can either use the total training set in one episode (train_len_d == eps_len_d) or
        divide the total training set into smaller subsets (train_len_d_i > eps_len_d). In the latter case, the
        subsets where selected randomly
        """

        np.random.seed(self.TrainConfig.seed_train)     # Set the random seed for random episode selection

        if self.EnvConfig.train_len_d == self.EnvConfig.eps_len_d:
            self.eps_ind = np.zeros(self.n_eps*int(self.num_loops)*self.overhead_factor)
        else:           # self.EnvConfig.train_len_d > self.EnvConfig.eps_len_d:
            # Random selection with sampling with replacement
            num_ep = np.linspace(start=0, stop=self.n_eps-1, num=self.n_eps)
            if self.num_loops < 1:      num_loops_int = 1
            else:                       num_loops_int = int(self.num_loops)
            random_ep = np.zeros((num_loops_int*self.overhead_factor, self.n_eps))
            for i in range(num_loops_int*self.overhead_factor):
                random_ep[i, :] = num_ep
                np.random.shuffle(random_ep[i, :])
            self.eps_ind = random_ep.reshape(int(self.n_eps*num_loops_int*self.overhead_factor)).astype(int)


    def dict_env_kwargs(self, type="train"):
        """
        Returns global model parameters and hyper parameters applied in the PtG environment as a dictionary
        :param dict_op_data: Dictionary with data of dynamic methanation operation
        :param type: Specifies either the training set "train" or the val/ test set "val_test"
        :return: env_kwargs: Dictionary with global parameters and hyperparameters
        """

        env_kwargs = {}
        ###################MAKE SHORTER #####################
        # More information on the environment's parameters are present in RL_PtG/src/rl_param_env.py

        env_kwargs["ptg_state_space['standby']"] = self.EnvConfig.ptg_state_space['standby'] 
        env_kwargs["ptg_state_space['cooldown']"] = self.EnvConfig.ptg_state_space['cooldown']
        env_kwargs["ptg_state_space['startup']"] = self.EnvConfig.ptg_state_space['startup']
        env_kwargs["ptg_state_space['partial_load']"] = self.EnvConfig.ptg_state_space['partial_load']
        env_kwargs["ptg_state_space['full_load']"] = self.EnvConfig.ptg_state_space['full_load']

        env_kwargs["noise"] = self.EnvConfig.noise
        env_kwargs["parallel"] = self.TrainConfig.parallel
        env_kwargs["eps_len_d"] = self.EnvConfig.eps_len_d
        env_kwargs["sim_step"] = self.EnvConfig.sim_step
        env_kwargs["time_step_op"] = self.EnvConfig.time_step_op
        env_kwargs["price_ahead"] = self.EnvConfig.price_ahead
        env_kwargs["n_eps_loops"] = self.n_eps_loops

        env_kwargs["dict_op_data['startup_cold']"] = self.dict_op_data['startup_cold']
        env_kwargs["dict_op_data['startup_hot']"] = self.dict_op_data['startup_hot']
        env_kwargs["dict_op_data['cooldown']"] = self.dict_op_data['cooldown']
        env_kwargs["dict_op_data['standby_down']"] = self.dict_op_data['standby_down']
        env_kwargs["dict_op_data['standby_up']"] = self.dict_op_data['standby_up']
        env_kwargs["dict_op_data['op1_start_p']"] = self.dict_op_data['op1_start_p']
        env_kwargs["dict_op_data['op2_start_f']"] = self.dict_op_data['op2_start_f']
        env_kwargs["dict_op_data['op3_p_f']"] = self.dict_op_data['op3_p_f']
        env_kwargs["dict_op_data['op4_p_f_p_5']"] = self.dict_op_data['op4_p_f_p_5']
        env_kwargs["dict_op_data['op5_p_f_p_10']"] = self.dict_op_data['op5_p_f_p_10']
        env_kwargs["dict_op_data['op6_p_f_p_15']"] = self.dict_op_data['op6_p_f_p_15']
        env_kwargs["dict_op_data['op7_p_f_p_22']"] = self.dict_op_data['op7_p_f_p_22']
        env_kwargs["dict_op_data['op8_f_p']"] = self.dict_op_data['op8_f_p']
        env_kwargs["dict_op_data['op9_f_p_f_5']"] = self.dict_op_data['op9_f_p_f_5']
        env_kwargs["dict_op_data['op10_f_p_f_10']"] = self.dict_op_data['op10_f_p_f_10']
        env_kwargs["dict_op_data['op11_f_p_f_15']"] = self.dict_op_data['op11_f_p_f_15']
        env_kwargs["dict_op_data['op12_f_p_f_20']"] = self.dict_op_data['op12_f_p_f_20']

        env_kwargs["scenario"] = self.EnvConfig.scenario

        env_kwargs["convert_mol_to_Nm3"] = self.EnvConfig.convert_mol_to_Nm3
        env_kwargs["H_u_CH4"] = self.EnvConfig.H_u_CH4
        env_kwargs["H_u_H2"] = self.EnvConfig.H_u_H2
        env_kwargs["dt_water"] = self.EnvConfig.dt_water
        env_kwargs["cp_water"] = self.EnvConfig.cp_water
        env_kwargs["rho_water"] = self.EnvConfig.rho_water
        env_kwargs["Molar_mass_CO2"] = self.EnvConfig.Molar_mass_CO2
        env_kwargs["Molar_mass_H2O"] = self.EnvConfig.Molar_mass_H2O
        env_kwargs["h_H2O_evap"] = self.EnvConfig.h_H2O_evap
        env_kwargs["eeg_el_price"] = self.EnvConfig.eeg_el_price
        env_kwargs["heat_price"] = self.EnvConfig.heat_price
        env_kwargs["o2_price"] = self.EnvConfig.o2_price
        env_kwargs["water_price"] = self.EnvConfig.water_price
        env_kwargs["min_load_electrolyzer"] = self.EnvConfig.min_load_electrolyzer
        env_kwargs["max_h2_volumeflow"] = self.EnvConfig.max_h2_volumeflow
        env_kwargs["eta_CHP"] = self.EnvConfig.eta_CHP

        env_kwargs["t_cat_standby"] = self.EnvConfig.t_cat_standby
        env_kwargs["t_cat_startup_cold"] = self.EnvConfig.t_cat_startup_cold
        env_kwargs["t_cat_startup_hot"] = self.EnvConfig.t_cat_startup_hot
        env_kwargs["time1_start_p_f"] = self.EnvConfig.time1_start_p_f
        env_kwargs["time2_start_f_p"] = self.EnvConfig.time2_start_f_p
        env_kwargs["time_p_f"] = self.EnvConfig.time_p_f
        env_kwargs["time_f_p"] = self.EnvConfig.time_f_p
        env_kwargs["time1_p_f_p"] = self.EnvConfig.time1_p_f_p
        env_kwargs["time2_p_f_p"] = self.EnvConfig.time2_p_f_p
        env_kwargs["time23_p_f_p"] = self.EnvConfig.time23_p_f_p
        env_kwargs["time3_p_f_p"] = self.EnvConfig.time3_p_f_p
        env_kwargs["time34_p_f_p"] = self.EnvConfig.time34_p_f_p
        env_kwargs["time4_p_f_p"] = self.EnvConfig.time4_p_f_p
        env_kwargs["time45_p_f_p"] = self.EnvConfig.time45_p_f_p
        env_kwargs["time5_p_f_p"] = self.EnvConfig.time5_p_f_p
        env_kwargs["time1_f_p_f"] = self.EnvConfig.time1_f_p_f
        env_kwargs["time2_f_p_f"] = self.EnvConfig.time2_f_p_f
        env_kwargs["time23_f_p_f"] = self.EnvConfig.time23_f_p_f
        env_kwargs["time3_f_p_f"] = self.EnvConfig.time3_f_p_f
        env_kwargs["time34_f_p_f"] = self.EnvConfig.time34_f_p_f
        env_kwargs["time4_f_p_f"] = self.EnvConfig.time4_f_p_f
        env_kwargs["time45_f_p_f"] = self.EnvConfig.time45_f_p_f
        env_kwargs["time5_f_p_f"] = self.EnvConfig.time5_f_p_f
        env_kwargs["i_fully_developed"] = self.EnvConfig.i_fully_developed
        env_kwargs["j_fully_developed"] = self.EnvConfig.j_fully_developed

        env_kwargs["t_cat_startup_cold"] = self.EnvConfig.t_cat_startup_cold
        env_kwargs["t_cat_startup_hot"] = self.EnvConfig.t_cat_startup_hot

        env_kwargs["T_l_b"] = self.EnvConfig.T_l_b
        env_kwargs["T_u_b"] = self.EnvConfig.T_u_b
        env_kwargs["h2_l_b"] = self.EnvConfig.h2_l_b
        env_kwargs["h2_u_b"] = self.EnvConfig.h2_u_b
        env_kwargs["ch4_l_b"] = self.EnvConfig.ch4_l_b
        env_kwargs["ch4_u_b"] = self.EnvConfig.ch4_u_b
        env_kwargs["h2_res_l_b"] = self.EnvConfig.h2_res_l_b
        env_kwargs["h2_res_u_b"] = self.EnvConfig.h2_res_u_b
        env_kwargs["h2o_l_b"] = self.EnvConfig.h2o_l_b
        env_kwargs["h2o_u_b"] = self.EnvConfig.h2o_u_b
        env_kwargs["heat_l_b"] = self.EnvConfig.heat_l_b
        env_kwargs["heat_u_b"] = self.EnvConfig.heat_u_b

        if type == "train":
            env_kwargs["eps_ind"] = self.eps_ind
            env_kwargs["state_change_penalty"] = self.EnvConfig.state_change_penalty
            env_kwargs["eps_sim_steps"] = self.eps_sim_steps_train
            env_kwargs["e_r_b"] = self.e_r_b_train                         
            env_kwargs["g_e"] = self.g_e_train 
            env_kwargs["rew_l_b"] = np.min(self.e_r_b_train[1, 0, :]) 
            env_kwargs["rew_u_b"] = np.max(self.e_r_b_train[1, 0, :])                            
        else:   # Evaluation 
            env_kwargs["eps_ind"] = np.zeros(len(self.eps_ind), dtype=int)          # Simply use the one validation or test set, no indexing required
            env_kwargs["state_change_penalty"] = 0.0        # no state change penalty during validation
            if type == "val":    # Validation
                env_kwargs["eps_sim_steps"] = self.eps_sim_steps_val
                env_kwargs["e_r_b"] = self.e_r_b_val                        
                env_kwargs["g_e"] = self.g_e_val
                env_kwargs["rew_l_b"] = np.min(self.e_r_b_val[1, 0, :]) 
                env_kwargs["rew_u_b"] = np.max(self.e_r_b_val[1, 0, :])  
            elif type == "test":    # Testing
                env_kwargs["eps_sim_steps"] = self.eps_sim_steps_test
                env_kwargs["e_r_b"] = self.e_r_b_test                         
                env_kwargs["g_e"] = self.g_e_test
                env_kwargs["rew_l_b"] = np.min(self.e_r_b_test[1, 0, :]) 
                env_kwargs["rew_u_b"] = np.max(self.e_r_b_test[1, 0, :])  
            else:
                assert False, f'The type argument ({type}) of dict_env_kwargs() needs to be "train" (training), "val" (validation) or "test" (testing)!'

        env_kwargs["reward_level"] = self.r_level
        env_kwargs["action_type"] = self.AgentConfig.rl_alg_hyp["action_type"]     # Action type of the algorithm, discrete or continuous
        env_kwargs["raw_modified"] = self.EnvConfig.raw_modified     # Specifies the type of state design using raw energy market prices ('raw') or modified economic metrices ('mod')

        return env_kwargs
    
def initial_print():
    print('\n--------------------------------------------------------------------------------------------')    
    print('---------RL_PtG: Deep Reinforcement Learning for Power-to-Gas dispatch optimization---------')
    print('--------------------------------------------------------------------------------------------\n')

def config_print(AgentConfig, EnvConfig, TrainConfig):
    """
        Aggregates and prints general settings
        :param AgentConfig: Agent configuration in a class object
        :param EnvConfig: Environment configuration in a class object
        :param TrainConfig: Training configuration in a class object
        :return str_id: String for identification of the present training run
    """
    print("Set training case...")
    if TrainConfig.model_conf == "simple_train" or TrainConfig.model_conf == "save_model":
        print(f"---Training case details: RL_PtG{TrainConfig.str_inv} | {TrainConfig.model_conf} ")
    else: 
        print(f"---Training case details: RL_PtG{TrainConfig.str_inv} | {TrainConfig.model_conf} | (Pretrained model: RL_PtG{TrainConfig.str_inv_load}) ")
    str_id = "RL_PtG_" + TrainConfig.str_inv
    if EnvConfig.scenario == 1: print("    > Business case (_BS):\t\t\t 1 - Trading at the electricity, gas, and emission spot markets")
    elif EnvConfig.scenario == 2: print("    > Business case  (_BS):\t\t\t 2 - Fixed synthetic natural gas (SNG) price and trading at the electricity and emission spot markets")
    else: print("    > Business case (_BS):\t\t\t 3 - Participating in EEG tenders by using a CHP plant and trading at the electricity spot markets")
    str_id += "_BS" + str(EnvConfig.scenario)
    print(f"    > Operational load level (_OP) :\t\t {EnvConfig.operation}")
    str_id += "_" + str(EnvConfig.operation)
    if EnvConfig.raw_modified == 'raw':    print(f"    > State feature design (_sf) :\t\t Raw energy market data (electricity, gas, and EUA market signals)")
    else: print(f"    > State feature design (_sf) :\t\t Modified energy market data (potential reward and load identifier)")
    str_id += "_sf" + str(EnvConfig.raw_modified)
    print(f"    > Training episode length (_ep) :\t\t {EnvConfig.eps_len_d} days")
    str_id += "_ep" + str(EnvConfig.eps_len_d)
    print(f"    > Time step size (action frequency) (_ts) :\t {EnvConfig.sim_step} seconds")
    str_id += "_ts" + str(EnvConfig.sim_step)
    print(f"    > Random seed (_rs) :\t\t\t {TrainConfig.seed_train}")
    str_id += AgentConfig.get_hyper()
    str_id += "_rs" + str(TrainConfig.seed_train)                       # Random seed at the end of "str_id" for file simplified file

    return str_id

def multiple_plots(stats_dict: dict, time_step_size: int, plot_name: str):
    """
    Creates a plot with multiple subplots of the time series and the methanation operation according to the agent
    :param stats_dict: dictionary with the prediction results
    :param time_step_size: time step size in the simulation
    :param plot_name: plot title
    :return
    """

    part_full_b = stats_dict['partial_full_b']
    time_sim = stats_dict['steps_stats'] * time_step_size * 1 / 3600 / 24  # in days
    time_sim_profit =  np.linspace(0, int(len(part_full_b))-1, num=int(len(part_full_b))) / 24
    profit_series = part_full_b * 50

    fig, axs = plt.subplots(8, 1, figsize=(10, 6), sharex=True, sharey=False)
    axs[0].plot(time_sim, stats_dict['el_price_stats'], label='el_price')
    axs[0].plot(time_sim, stats_dict['gas_price_stats'], 'g', label='gas_price')
    axs[0].plot(time_sim_profit, profit_series, 'r', label='profitable')
    # axs[0].set_ylim([0, 5.5])
    axs[0].set_ylabel('ct/kWh')
    axs[0].legend(loc="upper right", fontsize='x-small')
    axs0_1 = axs[0].twinx()
    axs0_1.plot(time_sim, stats_dict['eua_price_stats'], 'k', label='eua_price')
    axs0_1.set_ylabel('eua_price [€/t]')
    axs0_1.legend(loc="lower right", fontsize='x-small')
    axs[1].plot(time_sim, stats_dict['Meth_State_stats'], 'b', label='state')
    axs[1].plot(time_sim, stats_dict['Meth_Action_stats'], 'g', label='action')
    axs[1].plot(time_sim, stats_dict['Meth_Hot_Cold_stats'], 'k', label='hot/cold-status')
    # axs[1].set_ylim([0, 12])
    axs[1].set_ylabel('status')
    axs[1].legend(loc="upper right", fontsize='x-small')
    axs[2].plot(time_sim, stats_dict['Meth_T_cat_stats'], 'k', label='T_Cat')
    # axs[2].plot(time_sim, stats_dict['Meth_T_cat_stats'], 'k', marker='o', markersize=2)
    # axs[2].set_ylim([0, 600])
    axs[2].set_ylabel('°C')
    axs[2].legend(loc="upper right", fontsize='x-small')
    axs[3].plot(time_sim, stats_dict['Meth_H2_flow_stats'], 'b', label='H2')
    axs[3].plot(time_sim, stats_dict['Meth_CH4_flow_stats'], 'g', label='CH4')
    # axs[3].set_ylim([0, 0.025])
    axs[3].set_ylabel('mol/s')
    axs[3].legend(loc="upper right", fontsize='x-small')
    axs[4].plot(time_sim, stats_dict['Meth_H2O_flow_stats'], label='H2O')
    # axs[4].set_ylim([0, 0.72])
    axs[4].set_ylabel('kg/h')
    axs[4].legend(loc="upper right", fontsize='x-small')
    axs[5].plot(time_sim, stats_dict['Meth_el_heating_stats'], label='P_el_heat')
    # axs[5].set_ylim([-10, 2000])
    axs[5].set_ylabel('W')
    axs[5].legend(loc="upper right", fontsize='x-small')
    axs[6].plot(time_sim, stats_dict['Meth_ch4_revenues_stats'], 'g', label='CH4')
    axs[6].plot(time_sim, stats_dict['Meth_steam_revenues_stats'], 'b', label='H2O')
    axs[6].plot(time_sim, stats_dict['Meth_o2_revenues_stats'], 'lightgray', label='O2')
    axs[6].plot(time_sim, stats_dict['Meth_elec_costs_heating_stats'], 'k', label='P_el_heating')
    axs[6].plot(time_sim, stats_dict['Meth_elec_costs_electrolyzer_stats'], 'r', label='P_el_lyzer')
    axs[6].set_ylabel('ct/h')
    axs[6].legend(loc="upper right", fontsize='x-small')
    axs[7].plot(time_sim, stats_dict['Meth_reward_stats'], 'g', label='Reward')
    axs[7].set_ylabel('r [ct/h]')
    axs[7].set_xlabel('Time [d]')
    # axs[7].set_xlim([-10, 10000])
    axs[7].legend(loc="upper right", fontsize='x-small')
    axs7_1 = axs[7].twinx()
    axs7_1.plot(time_sim, stats_dict['Meth_cum_reward_stats'], 'k', label='Cumulative Reward')
    axs7_1.set_ylabel('cum_r [ct]')
    axs7_1.legend(loc="lower right", fontsize='x-small')

    fig.suptitle(" Alg:" + plot_name + "\n Rew:" + str(np.round(stats_dict['Meth_cum_reward_stats'][-1], 0)))
    plt.savefig('plots/' + plot_name + '_plot.png')
    print("Reward =", stats_dict['Meth_cum_reward_stats'][-1])

    # plt.show()
    plt.close()


def multiple_plots_4(stats_dict: dict, time_step_size: int, plot_name: str, potential_profit: np.array, part_full: np.array):
    """
    Creates a plot with multiple subplots of the time series and the methanation operation according to the agent
    :param stats_dict: dictionary with the prediction results
    :param time_step_size: time step size in the simulation
    :param plot_name: plot title
    :param profit: denotes periods with potential profit (No Profit = 0, Profit = 10)
    :return:
    """

    time_sim = stats_dict['steps_stats'] * time_step_size * 1 / 3600 / 24  # in days
    time_sim_profit =  np.linspace(0, int(len(profit))-1, num=int(len(profit))) / 24
    profit_series = 1+ profit * 4

    P_PtG = stats_dict['Meth_H2_flow_stats'] / 0.0198

    plt.rcParams.update({'font.size': 16})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    pot_rew = potential_profit * time_step_size / 3600

    rew_zero = np.zeros((len(time_sim),))
    part_full_adapted = np.zeros((len(time_sim),))
    for i in range(len(time_sim)):
        if part_full[i] == -1:
            part_full_adapted[i] = 2
        elif part_full[i] == 0:
            part_full_adapted[i] = 4
        else:
            part_full_adapted[i] = 5

    fig, axs = plt.subplots(1, 1, figsize=(14, 6), sharex=True, sharey=False)
    axs.plot(time_sim, stats_dict['Meth_State_stats'], 'b', label='state')
    axs.plot(time_sim, part_full_adapted, 'k', linestyle="dotted", label='state')
    axs.set_yticks([1,2,3,4,5])
    axs.set_yticklabels(['Standby', 'Cooldown/Off', 'Startup', 'Partial Load', 'Full Load'])
    # axs[1].set_ylim([0, 12])
    axs.set_ylabel(' ')
    axs.legend(loc="upper left", fontsize='small') #, bbox_to_anchor = (0.0, 0.0), ncol = 1, fancybox = True, shadow = True)
    axs.grid(axis='y', linestyle='dashed')
    axs0_1 = axs.twinx()
    axs0_1.plot(time_sim, stats_dict['Meth_reward_stats'], color='g', label='Reward')
    axs0_1.plot(time_sim, pot_rew, color='lawngreen', linestyle='dotted', label='Potential Reward')
    axs0_1.plot(time_sim, rew_zero, color='grey', linestyle='dashed')
    axs0_1.set_ylabel('reward')
    axs0_1.set_xlabel('Time [d]')
    axs0_1.set_yticks([0, 10, 20])
    # axs3_1 = axs[3].twinx()
    # axs3_1.plot(time_sim, stats_dict['Meth_cum_reward_stats'], 'k', label='Cumulative Reward')
    # axs3_1.set_ylabel('cum. reward')
    # # axs3_1.set_yticks([0, 1000, 2000])
    # axs[3].legend(loc="upper left", fontsize='small') #, bbox_to_anchor=(0.0, 0.0), ncol=1, fancybox=True, shadow=True)
    # axs3_1.legend(loc="upper right", fontsize='small') #, bbox_to_anchor = (0.0, 0.0), ncol = 1, fancybox = True, shadow = True)

    # box = axs0_1.get_position()
    # axs0_1.set_position([box.x0 * 1.1, box.y0 * 1.05, box.width, box.height])
    # box = axs[1].get_position()
    # axs[1].set_position([box.x0 * 1.1, box.y0 * 1.05, box.width, box.height])
    # box = axs2_1.get_position()
    # axs2_1.set_position([box.x0 * 1.1, box.y0 * 1.05, box.width, box.height])
    # box = axs3_1.get_position()
    # axs3_1.set_position([box.x0 * 1.1, box.y0, box.width, box.height])

    fig.suptitle(" Alg:" + plot_name + "\n Rew:" + str(np.round(stats_dict['Meth_cum_reward_stats'][-1], 0)))
    plt.savefig('plots/' + plot_name + '_plot.png')
    # print("Reward =", stats_dict['Meth_cum_reward_stats'][-1])

    plt.close()
    # plt.show()





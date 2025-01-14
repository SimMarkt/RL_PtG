import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import math

from src.rl_param_agent import AgentParams
from src.rl_param_env import EnvParams
from src.rl_param_train import TrainParams
from src.rl_opt import calculate_optimum

def import_market_data(csvfile: str, type: str):
    """
        Import data of day-ahead prices for electricity
        :param csvfile: Path and name of the .csv files containing energy market data ["Time [s]"; <data>]
        :param type: market data type
        :return arr: np.array with market data
    """

    file_path = os.path.dirname(__file__) + "/" + csvfile
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


def import_data(csvfile: str):
    """
        Import experimental methanation data for state changes
        :param csvfile: Path and name of the .csv files containing process data ["Time [s]", "T_cat [°C]", "n_h2 [mol/s]", "n_ch4 [mol/s]", "n_h2_res [mol/s]", "m_DE [kg/h]", "Pel [W]"]
        :return arr: Numpy array with operational data
    """
    # Import historic Data from csv file
    file_path = os.path.dirname(__file__) + "/" + csvfile
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


def load_data():
    """
        Loads historical market data and experimental data of methanation operation
        :return dict_price_data: dictionary with market data
                dict_op_data: dictionary with data of dynamic methanation operation
    """

    ENV_PARAMS = EnvParams()

    # Load historical market data for electricity, gas and EUA ##########################################MAKE SHORTER##################
    dict_price_data = {'el_price_train': import_market_data(ENV_PARAMS.datafile_path_train_el, "elec"),     # Electricity prices of the training set
                       'el_price_cv': import_market_data(ENV_PARAMS.datafile_path_cv_el, "elec"),           # Electricity prices of the validation set
                       'el_price_test': import_market_data(ENV_PARAMS.datafile_path_test_el, "elec"),       # Electricity prices of the test set
                       'gas_price_train': import_market_data(ENV_PARAMS.datafile_path_train_gas, "gas"),    # Gas prices of the training set
                       'gas_price_cv': import_market_data(ENV_PARAMS.datafile_path_cv_gas, "gas"),          # Gas prices of the validation set
                       'gas_price_test': import_market_data(ENV_PARAMS.datafile_path_test_gas, "gas"),      # Gas prices of the test set
                       'eua_price_train': import_market_data(ENV_PARAMS.datafile_path_train_eua, "eua"),    # EUA prices of the training set
                       'eua_price_cv': import_market_data(ENV_PARAMS.datafile_path_cv_eua, "eua"),          # EUA prices of the validation set
                       'eua_price_test': import_market_data(ENV_PARAMS.datafile_path_test_eua, "eua")}      # EUA prices of the test set

    # Load experimental methanation data for state changes  ##########################################MAKE SHORTER##################
    dict_op_data = {'startup_cold': import_data(ENV_PARAMS.datafile_path2),     # Cold start
                    'startup_hot': import_data(ENV_PARAMS.datafile_path3),      # Hot start
                    'cooldown': import_data(ENV_PARAMS.datafile_path4),         # Cooldown
                    'standby_down': import_data(ENV_PARAMS.datafile_path5),     # Standby dataset for high temperatures to standby
                    'standby_up': import_data(ENV_PARAMS.datafile_path6),       # Standby dataset for low temperatures to standby
                    'op1_start_p': import_data(ENV_PARAMS.datafile_path7),      # Partial load - warming up
                    'op2_start_f': import_data(ENV_PARAMS.datafile_path8),      # Full load - warming up
                    'op3_p_f': import_data(ENV_PARAMS.datafile_path9),          # Load change: Partial -> Full
                    'op4_p_f_p_5': import_data(ENV_PARAMS.datafile_path10),     # Load change: Partial -> Full -> Partial (Return after 5 min)
                    'op5_p_f_p_10': import_data(ENV_PARAMS.datafile_path11),    # Load change: Partial -> Full -> Partial (Return after 10 min)
                    'op6_p_f_p_15': import_data(ENV_PARAMS.datafile_path12),    # Load change: Partial -> Full -> Partial (Return after 15 min)
                    'op7_p_f_p_22': import_data(ENV_PARAMS.datafile_path13),    # Load change: Partial -> Full -> Partial (Return after 22 min)
                    'op8_f_p': import_data(ENV_PARAMS.datafile_path14),         # Load change: Full -> Partial
                    'op9_f_p_f_5': import_data(ENV_PARAMS.datafile_path15),     # Load change: Full -> Partial -> Full (Return after 5 min)
                    'op10_f_p_f_10': import_data(ENV_PARAMS.datafile_path16),   # Load change: Full -> Partial -> Full (Return after 10 min)
                    'op11_f_p_f_15': import_data(ENV_PARAMS.datafile_path17),   # Load change: Full -> Partial -> Full (Return after 15 min)
                    'op12_f_p_f_20': import_data(ENV_PARAMS.datafile_path18)}   # Load change: Full -> Partial -> Full (Return after 20 min)

    if ENV_PARAMS.scenario == 2:  # Fixed gas prices    ##########################################MAKE SHORTER##################
        dict_price_data['gas_price_train'] = np.ones(len(dict_price_data['gas_price_train'])) * ENV_PARAMS.ch4_price_fix
        dict_price_data['gas_price_cv'] = np.ones(len(dict_price_data['gas_price_cv'])) * ENV_PARAMS.ch4_price_fix
        dict_price_data['gas_price_test'] = np.ones(len(dict_price_data['gas_price_test'])) * ENV_PARAMS.ch4_price_fix
    elif ENV_PARAMS.scenario == 3:  # Gas and EUA prices = 0
        dict_price_data['gas_price_train'] = np.zeros(len(dict_price_data['gas_price_train']))
        dict_price_data['gas_price_cv'] = np.zeros(len(dict_price_data['gas_price_cv']))
        dict_price_data['gas_price_test'] = np.zeros(len(dict_price_data['gas_price_test']))
        dict_price_data['eua_price_train'] = np.zeros(len(dict_price_data['eua_price_train']))
        dict_price_data['eua_price_cv'] = np.zeros(len(dict_price_data['eua_price_test']))
        dict_price_data['eua_price_test'] = np.zeros(len(dict_price_data['eua_price_test']))

    # For Reward level calculation -> Sets height of the reward penalty
    dict_price_data['el_price_reward_level'] = ENV_PARAMS.r_0_values['el_price']
    dict_price_data['gas_price_reward_level'] = ENV_PARAMS.r_0_values['gas_price']
    dict_price_data['eua_price_reward_level'] = ENV_PARAMS.r_0_values['eua_price']

    # Check if training set is divisible by the episode length
    min_train_len = 5           # Minimum No. of days in the training set 
    ENV_PARAMS.train_len_d = len(dict_price_data['gas_price_train']) - min_train_len # Training uses min_train_len days less than the data size to always allow enough space for the price forecast of Day-ahead market data
    assert ENV_PARAMS.train_len_d > 0, f'The training set size must be greater than {min_train_len} days'
    if ENV_PARAMS.train_len_d % ENV_PARAMS.eps_len_d != 0:
        # Find all possible divisors of ENV_PARAMS.train_len_d
        divisors = [i for i in range(1, ENV_PARAMS.train_len_d + 1) if ENV_PARAMS.train_len_d % i == 0]
        assert False, f'The training set size {ENV_PARAMS.train_len_d} must be divisible by the episode length - data/config_env.yaml -> eps_len_d : {ENV_PARAMS.eps_len_d}; 
                        Possible divisors are: {divisors}'
    # Ensure that the training set is larger or at least equal to the defined episode length  
    assert ENV_PARAMS.train_len_d >= ENV_PARAMS.eps_len_d, f'Training set size ({ENV_PARAMS.train_len_d}) must be larger or at least equal to the defined episode length ({ENV_PARAMS.eps_len_d})!'

    return dict_price_data, dict_op_data

class Preprocessing():
    """
        A class that contains variables and functions for preprocessing of energy market and process data
    """
    def __init__(self, dict_price_data, seed_train):
        """
            Initialization of variables
            :param dict_price_data: dictionary with market data
            :param seed_train: Random seed of the training set
        """
        # Initialization
        self.AGENT_PARAMS = AgentParams()
        self.ENV_PARAMS = EnvParams()
        self.TRAIN_PARAMS = TrainParams()
        self.dict_price_data = dict_price_data
        self.seed_train = seed_train
        self.dict_pot_r_b = None                    # dictionary with potential reward [pot_rew...] and boolean reward identifier [part_full_b...]
        self.r_level = None                         # Sets the general height of the reward penalty according to electricity, (S)NG, and EUA price levels
        # e_r_b_train/e_r_b_cv/e_r_b_test: (hourly values)
        #   np.array which stores elec. price data, potential reward, and boolean identifier
        #   Dimensions = [Type of data] x [No. of day-ahead values] x [historical values]
        #       Type of data = [el_price, pot_rew, part_full_b]
        #       No. of day-ahead values = ENV_PARAMS.price_ahead
        #       historical values = No. of values in the electricity price data set
        self.e_r_b_train = None
        self.e_r_b_cv = None
        self.e_r_b_test = None
        # g_e_train/g_e_cv/g_e_test: (daily values)
        #   np.array which stores gas and EUA price data
        #   Dimensions = [Type of data] x [No. of day-ahead values] x [historical values]
        #       Type of data = [gas_price, pot_rew, part_full_b]
        #       No. of day-ahead values = 2 (today and tomorrow)
        #       historical values = No. of values in the gas/EUA price data set
        self.g_e_train = None
        self.g_e_cv = None
        self.g_e_test = None
        # Variables for division of the entire training set into different, randomly picked subsets for episodic learing
        self.eps_sim_steps_train = None         # Number of steps in the training set per episode
        self.eps_sim_steps_test = None          # Number of steps in the test set
        self.eps_ind = None                     # Contains indexes of the randomly ordered training subsets
        self.overhead_factor = 3                # Overhead of self.eps_ind - To account for randomn selection of the different processes in multiprocessing
        self.n_eps = None                       # Episode length in seconds
        self.num_loops = None                   # No. of loops over the total training set during training

        # For Multiprocessing: self.n_eps_loops allows for definition of different eps_ind for different processes (see RL_PtG\env\ptg_gym_env.py)
        self.n_eps_loops = None                 # Total No. of episodes over the entire training procedure 
        

    def preprocessing_rew(self):
        """
            Data preprocessing including the computation of a potential reward, which signifies the maximum reward the
            Power-to-Gas plant can yield in either partial load [part_full_b... = 0] or full load [part_full_b... = 1]
            :return dict_pot_r_b: dictionary with potential reward [pot_rew...] and boolean reward identifier [part_full_b...]
            :return r_level: ###################################################################################################################################?????????
        """

        # Compute methanation operation data for theoretical optimum (ignoring dynamics) ##########################################MAKE SHORTER##################
        # calculate_optimum() had been excluded from Preprocessing() and placed in the different rl_opt.py file for the sake of clarity
        stats_dict_opt_train = calculate_optimum(self.dict_price_data['el_price_train'], self.dict_price_data['gas_price_train'],
                                                self.dict_price_data['eua_price_train'], "Train")
        stats_dict_opt_cv = calculate_optimum(self.dict_price_data['el_price_cv'], self.dict_price_data['gas_price_cv'],
                                                self.dict_price_data['eua_price_cv'], "CV")
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
            'pot_rew_cv': stats_dict_opt_cv['Meth_reward_stats'],
            'part_full_b_cv': stats_dict_opt_cv['partial_full_b'],
            'pot_rew_test': stats_dict_opt_test['Meth_reward_stats'],
            'part_full_b_test': stats_dict_opt_test['partial_full_b'],
        }

        self.r_level = stats_dict_opt_level['Meth_reward_stats']
        # multiple_plots(stats_dict_opt_train, 3600, "Opt_Training_set_sen" + str(ENV_PARAMS.scenario))         ################# Include Graphics as default ##############
        # multiple_plots(stats_dict_opt_test, 3600, "Opt_Test_set_sen" + str(ENV_PARAMS.scenario))              ################# Include Graphics as default ##############


    def preprocessing_array(self):
        """
        Transforms dictionaries to np.arrays for computational purposes
        """

        # Multi-Dimensional Array (3D) which stores day-ahead electricity price data as well as day-ahead potential reward
        # and boolean identifier for the entire training and test set
        # e.g. e_r_b_train[0, 5, 156] represents the future value of the electricity price [0,-,-] in 4 hours [-,5,-] at the
        # 156ths entry of the electricity price data set             ##########################################MAKE SHORTER##################
        e_r_b_train = np.zeros((3, self.ENV_PARAMS.price_ahead, self.dict_price_data['el_price_train'].shape[0] - self.ENV_PARAMS.price_ahead))
        e_r_b_cv = np.zeros((3, self.ENV_PARAMS.price_ahead, self.dict_price_data['el_price_cv'].shape[0] - self.ENV_PARAMS.price_ahead))
        e_r_b_test = np.zeros((3, self.ENV_PARAMS.price_ahead, self.dict_price_data['el_price_test'].shape[0] - self.ENV_PARAMS.price_ahead))

        for i in range(self.ENV_PARAMS.price_ahead):     ##########################################MAKE SHORTER##################
            e_r_b_train[0, i, :] = self.dict_price_data['el_price_train'][i:(-self.ENV_PARAMS.price_ahead + i)]
            e_r_b_train[1, i, :] = self.dict_pot_r_b['pot_rew_train'][i:(-self.ENV_PARAMS.price_ahead + i)]
            e_r_b_train[2, i, :] = self.dict_pot_r_b['part_full_b_train'][i:(-self.ENV_PARAMS.price_ahead + i)]
            e_r_b_cv[0, i, :] = self.dict_price_data['el_price_cv'][i:(-self.ENV_PARAMS.price_ahead + i)]
            e_r_b_cv[1, i, :] = self.dict_pot_r_b['pot_rew_cv'][i:(-self.ENV_PARAMS.price_ahead + i)]
            e_r_b_cv[2, i, :] = self.dict_pot_r_b['part_full_b_cv'][i:(-self.ENV_PARAMS.price_ahead + i)]
            e_r_b_test[0, i, :] = self.dict_price_data['el_price_test'][i:(-self.ENV_PARAMS.price_ahead + i)]
            e_r_b_test[1, i, :] = self.dict_pot_r_b['pot_rew_test'][i:(-self.ENV_PARAMS.price_ahead + i)]
            e_r_b_test[2, i, :] = self.dict_pot_r_b['part_full_b_test'][i:(-self.ENV_PARAMS.price_ahead + i)]

        # Multi-Dimensional Array (3D) which stores day-ahead gas and eua price data for the entire training and test set        ##########################################MAKE SHORTER##################
        g_e_train = np.zeros((2, 2, self.dict_price_data['gas_price_train'].shape[0] - 1))
        g_e_cv = np.zeros((2, 2, self.dict_price_data['gas_price_cv'].shape[0] - 1))
        g_e_test = np.zeros((2, 2, self.dict_price_data['gas_price_test'].shape[0] - 1))

        g_e_train[0, 0, :] = self.dict_price_data['gas_price_train'][:-1]     ##########################################MAKE SHORTER##################
        g_e_train[1, 0, :] = self.dict_price_data['eua_price_train'][:-1]
        g_e_cv[0, 0, :] = self.dict_price_data['gas_price_cv'][:-1]
        g_e_cv[1, 0, :] = self.dict_price_data['eua_price_cv'][:-1]
        g_e_test[0, 0, :] = self.dict_price_data['gas_price_test'][:-1]
        g_e_test[1, 0, :] = self.dict_price_data['eua_price_test'][:-1]
        g_e_train[0, 1, :] = self.dict_price_data['gas_price_train'][1:]
        g_e_train[1, 1, :] = self.dict_price_data['eua_price_train'][1:]
        g_e_cv[0, 1, :] = self.dict_price_data['gas_price_cv'][1:]
        g_e_cv[1, 1, :] = self.dict_price_data['eua_price_cv'][1:]
        g_e_test[0, 1, :] = self.dict_price_data['gas_price_test'][1:]
        g_e_test[1, 1, :] = self.dict_price_data['eua_price_test'][1:]


    def define_episodes(self):
        """
        Defines specifications for training and evaluation episodes
        """

        print("Define episodes and step size limits...")
        # No. of days in the test set ("-1" excludes the day-ahead overhead)
        cv_len_d = len(self.dict_price_data['gas_price_cv']) - 1
        test_len_d = len(self.dict_price_data['gas_price_test']) - 1

        # Split up the entire training set into several smaller subsets which represents an own episodes
        self.n_eps = int(self.ENV_PARAMS.train_len_d / self.ENV_PARAMS.eps_len_d)  # Number of training subsets/episodes per training procedure
        
        self.eps_len = 24 * 3600 * self.ENV_PARAMS.eps_len_d  # episode length in seconds

        # Number of steps in train and test sets per episode
        self.eps_sim_steps_train = int(self.eps_len / self.ENV_PARAMS.sim_step)
        self.eps_sim_steps_cv = int(24 * 3600 * cv_len_d / self.ENV_PARAMS.sim_step)
        self.eps_sim_steps_test = int(24 * 3600 * test_len_d /self. ENV_PARAMS.sim_step)

        # Define total number of steps for all workers together
        self.num_loops = int(self.TRAIN_PARAMS.total_steps / (self.eps_sim_steps_train * self.n_eps))  # Number of loops over the total training set
        print("--- Total number of training steps =", self.TRAIN_PARAMS.total_steps)
        print("--- No. of loops over the entire training set =", self.num_loops)
        print("--- Training steps per episode =", self.eps_sim_steps_train)
        print("--- Steps in the evaluation set =", self.eps_sim_steps_test)

        # Create random selection routine with replacement for the different training subsets
        self.rand_eps_ind()

        # For Multiprocessing, eps_ind should not shared between different processes
        self.n_eps_loops = self.n_eps * self.num_loops  # Allows for definition of different eps_ind in Multiprocessing (see RL_PtG\env\ptg_gym_env.py)


    def rand_eps_ind(self):
        """
        The agent can either use the total training set in one episode (train_len_d == eps_len_d) or
        divide the total training set into smaller subsets (train_len_d_i > eps_len_d). In the latter case, the
        subsets where selected randomly
        """

        np.random.seed(self.seed_train)     # Set the random seed for random episode selection

        if self.ENV_PARAMS.train_len_d == self.ENV_PARAMS.eps_len_d:
            self.eps_ind = np.zeros(int(self.n_eps*self.num_loops*self.overhead_factor))
        else:           # self.ENV_PARAMS.train_len_d > self.ENV_PARAMS.eps_len_d:
            # Random selection with sampling with replacement
            num_ep = np.linspace(start=0, stop=self.n_eps-1, num=self.n_eps)
            random_ep = np.zeros((self.num_loops*self.overhead_factor, self.n_eps))
            for i in range(self.num_loops*self.overhead_factor):
                random_ep[i, :] = num_ep
                np.random.shuffle(random_ep[i, :])
            self.eps_ind = random_ep.reshape(int(self.n_eps*self.num_loops*self.overhead_factor)).astype(int)


    def dict_env_kwargs(self, dict_op_data, type="train"):
        """
        Returns global model parameters and hyper parameters applied in the PtG environment as a dictionary
        :param dict_op_data: Dictionary with data of dynamic methanation operation
        :param type: Specifies either the training set "train" or the cv/ test set "cv_test"
        :return: env_kwargs: Dictionary with global parameters and hyperparameters
        """

        env_kwargs = {}

        env_kwargs["ptg_state_space['standby']"] = self.ENV_PARAMS.ptg_state_space['standby'] ###################MAKE SHORTER #####################
        env_kwargs["ptg_state_space['cooldown']"] = self.ENV_PARAMS.ptg_state_space['cooldown']
        env_kwargs["ptg_state_space['startup']"] = self.ENV_PARAMS.ptg_state_space['startup']
        env_kwargs["ptg_state_space['partial_load']"] = self.ENV_PARAMS.ptg_state_space['partial_load']
        env_kwargs["ptg_state_space['full_load']"] = self.ENV_PARAMS.ptg_state_space['full_load']

        env_kwargs["noise"] = self.ENV_PARAMS.noise
        env_kwargs["parallel"] = self.TRAIN_PARAMS.parallel
        env_kwargs["eps_len_d"] = self.ENV_PARAMS.eps_len_d
        env_kwargs["sim_step"] = self.ENV_PARAMS.sim_step
        env_kwargs["time_step_op"] = self.ENV_PARAMS.time_step_op
        env_kwargs["price_ahead"] = self.ENV_PARAMS.price_ahead
        env_kwargs["n_eps_loops"] = self.n_eps_loops

        env_kwargs["eps_ind"] = self.eps_ind                     # differ in train and test set
        env_kwargs["e_r_b"] = self.e_r_b                         # differ in train and test set
        env_kwargs["g_e"] = self.g_e                             # differ in train and test set

        env_kwargs["dict_op_data['startup_cold']"] = dict_op_data['startup_cold']
        env_kwargs["dict_op_data['startup_hot']"] = dict_op_data['startup_hot']
        env_kwargs["dict_op_data['cooldown']"] = dict_op_data['cooldown']
        env_kwargs["dict_op_data['standby_down']"] = dict_op_data['standby_down']
        env_kwargs["dict_op_data['standby_up']"] = dict_op_data['standby_up']
        env_kwargs["dict_op_data['op1_start_p']"] = dict_op_data['op1_start_p']
        env_kwargs["dict_op_data['op2_start_f']"] = dict_op_data['op2_start_f']
        env_kwargs["dict_op_data['op3_p_f']"] = dict_op_data['op3_p_f']
        env_kwargs["dict_op_data['op4_p_f_p_5']"] = dict_op_data['op4_p_f_p_5']
        env_kwargs["dict_op_data['op5_p_f_p_10']"] = dict_op_data['op5_p_f_p_10']
        env_kwargs["dict_op_data['op6_p_f_p_15']"] = dict_op_data['op6_p_f_p_15']
        env_kwargs["dict_op_data['op7_p_f_p_22']"] = dict_op_data['op7_p_f_p_22']
        env_kwargs["dict_op_data['op8_f_p']"] = dict_op_data['op8_f_p']
        env_kwargs["dict_op_data['op9_f_p_f_5']"] = dict_op_data['op9_f_p_f_5']
        env_kwargs["dict_op_data['op10_f_p_f_10']"] = dict_op_data['op10_f_p_f_10']
        env_kwargs["dict_op_data['op11_f_p_f_15']"] = dict_op_data['op11_f_p_f_15']
        env_kwargs["dict_op_data['op12_f_p_f_20']"] = dict_op_data['op12_f_p_f_20']

        env_kwargs["scenario"] = ENV_PARAMS.scenario

        env_kwargs["convert_mol_to_Nm3"] = ENV_PARAMS.convert_mol_to_Nm3
        env_kwargs["H_u_CH4"] = ENV_PARAMS.H_u_CH4
        env_kwargs["H_u_H2"] = ENV_PARAMS.H_u_H2
        env_kwargs["dt_water"] = ENV_PARAMS.dt_water
        env_kwargs["cp_water"] = ENV_PARAMS.cp_water
        env_kwargs["rho_water"] = ENV_PARAMS.rho_water
        env_kwargs["Molar_mass_CO2"] = ENV_PARAMS.Molar_mass_CO2
        env_kwargs["Molar_mass_H2O"] = ENV_PARAMS.Molar_mass_H2O
        env_kwargs["h_H2O_evap"] = ENV_PARAMS.h_H2O_evap
        env_kwargs["eeg_el_price"] = ENV_PARAMS.eeg_el_price
        env_kwargs["heat_price"] = ENV_PARAMS.heat_price
        env_kwargs["o2_price"] = ENV_PARAMS.o2_price
        env_kwargs["water_price"] = ENV_PARAMS.water_price
        env_kwargs["min_load_electrolyzer"] = ENV_PARAMS.min_load_electrolyzer
        env_kwargs["max_h2_volumeflow"] = ENV_PARAMS.max_h2_volumeflow
        env_kwargs["eta_BHKW"] = ENV_PARAMS.eta_BHKW

        env_kwargs["t_cat_standby"] = ENV_PARAMS.t_cat_standby
        env_kwargs["t_cat_startup_cold"] = ENV_PARAMS.t_cat_startup_cold
        env_kwargs["t_cat_startup_hot"] = ENV_PARAMS.t_cat_startup_hot
        env_kwargs["time1_start_p_f"] = ENV_PARAMS.time1_start_p_f
        env_kwargs["time2_start_f_p"] = ENV_PARAMS.time2_start_f_p
        env_kwargs["time_p_f"] = ENV_PARAMS.time_p_f
        env_kwargs["time_f_p"] = ENV_PARAMS.time_f_p
        env_kwargs["time1_p_f_p"] = ENV_PARAMS.time1_p_f_p
        env_kwargs["time2_p_f_p"] = ENV_PARAMS.time2_p_f_p
        env_kwargs["time23_p_f_p"] = ENV_PARAMS.time23_p_f_p
        env_kwargs["time3_p_f_p"] = ENV_PARAMS.time3_p_f_p
        env_kwargs["time34_p_f_p"] = ENV_PARAMS.time34_p_f_p
        env_kwargs["time4_p_f_p"] = ENV_PARAMS.time4_p_f_p
        env_kwargs["time45_p_f_p"] = ENV_PARAMS.time45_p_f_p
        env_kwargs["time5_p_f_p"] = ENV_PARAMS.time5_p_f_p
        env_kwargs["time1_f_p_f"] = ENV_PARAMS.time1_f_p_f
        env_kwargs["time2_f_p_f"] = ENV_PARAMS.time2_f_p_f
        env_kwargs["time23_f_p_f"] = ENV_PARAMS.time23_f_p_f
        env_kwargs["time3_f_p_f"] = ENV_PARAMS.time3_f_p_f
        env_kwargs["time34_f_p_f"] = ENV_PARAMS.time34_f_p_f
        env_kwargs["time4_f_p_f"] = ENV_PARAMS.time4_f_p_f
        env_kwargs["time45_f_p_f"] = ENV_PARAMS.time45_f_p_f
        env_kwargs["time5_f_p_f"] = ENV_PARAMS.time5_f_p_f
        env_kwargs["i_fully_developed"] = ENV_PARAMS.i_fully_developed
        env_kwargs["j_fully_developed"] = ENV_PARAMS.j_fully_developed

        env_kwargs["t_cat_startup_cold"] = ENV_PARAMS.t_cat_startup_cold
        env_kwargs["t_cat_startup_hot"] = ENV_PARAMS.t_cat_startup_hot

        env_kwargs["rew_l_b"] = np.min(e_r_b[1, 0, :])
        env_kwargs["rew_u_b"] = np.max(e_r_b[1, 0, :])
        env_kwargs["T_l_b"] = ENV_PARAMS.T_l_b
        env_kwargs["T_u_b"] = ENV_PARAMS.T_u_b
        env_kwargs["h2_l_b"] = ENV_PARAMS.h2_l_b
        env_kwargs["h2_u_b"] = ENV_PARAMS.h2_u_b
        env_kwargs["ch4_l_b"] = ENV_PARAMS.ch4_l_b
        env_kwargs["ch4_u_b"] = ENV_PARAMS.ch4_u_b
        env_kwargs["h2_res_l_b"] = ENV_PARAMS.h2_res_l_b
        env_kwargs["h2_res_u_b"] = ENV_PARAMS.h2_res_u_b
        env_kwargs["h2o_l_b"] = ENV_PARAMS.h2o_l_b
        env_kwargs["h2o_u_b"] = ENV_PARAMS.h2o_u_b
        env_kwargs["heat_l_b"] = ENV_PARAMS.heat_l_b
        env_kwargs["heat_u_b"] = ENV_PARAMS.heat_u_b

        env_kwargs["eps_sim_steps"] = eps_sim_steps         # differ in train and test set

        if type == "train":
            env_kwargs["state_change_penalty"] = AGENT_PARAMS.state_change_penalty
        elif type == "cv_test":
            env_kwargs["state_change_penalty"] = 0.0        # no state change penalty during validation

        env_kwargs["reward_level"] = r_level
        env_kwargs["action_type"] = ENV_PARAMS.action_type

        return env_kwargs


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


def get_param_str(eps_sim_steps_train, seed):                           ################################------------------------------------------------------------
    """
    Returns the hyperparameter setting as a long string
    :return: hyperparameter setting
    """
    AGENT_PARAMS = AgentParams()

    str_params_short = "_ep" + str(AGENT_PARAMS.eps_len_d) + \
                       "_al" + str(np.round(AGENT_PARAMS.alpha, 6)) + \
                       "_ga" + str(np.round(AGENT_PARAMS.gamma, 4)) + \
                       "_bt" + str(AGENT_PARAMS.batch_size) + \
                       "_bf" + str(AGENT_PARAMS.buffer_size) + \
                       "_et" + str(np.round(AGENT_PARAMS.ent_coeff, 5)) + \
                       "_hu" + str(AGENT_PARAMS.hidden_units) + \
                       "_hl" + str(AGENT_PARAMS.hidden_layers) + \
                       "_st" + str(AGENT_PARAMS.sim_step) + \
                       "_ac" + str(AGENT_PARAMS.activation) + \
                       "_ls" + str(AGENT_PARAMS.learning_starts) + \
                       "_tf" + str(AGENT_PARAMS.train_freq) + \
                       "_tau" + str(AGENT_PARAMS.tau) + \
                       "_cr" + str(AGENT_PARAMS.n_critics) + \
                       "_qu" + str(AGENT_PARAMS.n_quantiles) + \
                       "_qd" + str(AGENT_PARAMS.top_quantiles_drop) + \
                       "_gsd" + str(AGENT_PARAMS.gSDE) + \
                       "_sd" + str(seed)

    str_params_long = "\n     episode length=" + str(AGENT_PARAMS.eps_len_d) + \
                      "\n     alpha=" + str(AGENT_PARAMS.alpha) + \
                      "\n     gamma=" + str(AGENT_PARAMS.gamma) + \
                      "\n     batchsize=" + str(AGENT_PARAMS.batch_size) + \
                      "\n     replaybuffer=" + str(AGENT_PARAMS.buffer_size) + \
                      " (#ofEpisodes=" + str(AGENT_PARAMS.buffer_size / eps_sim_steps_train) + ")" + \
                      "\n     coeff_ent=" + str(AGENT_PARAMS.ent_coeff) + \
                      "\n     hiddenunits=" + str(AGENT_PARAMS.hidden_units) + \
                      "\n     hiddenlayers=" + str(AGENT_PARAMS.hidden_layers) + \
                      "\n     sim_step=" + str(AGENT_PARAMS.sim_step) + \
                      "\n     activation=" + str(AGENT_PARAMS.activation) + " (0=Relu, 1=tanh" + ")" + \
                      "\n     learningstarts=" + str(AGENT_PARAMS.learning_starts) + \
                      "\n     training_freq=" + str(AGENT_PARAMS.train_freq) + \
                      "\n     tau=" + str(AGENT_PARAMS.tau) + \
                      "\n     n_critics=" + str(AGENT_PARAMS.n_critics) + \
                      "\n     n_quantiles=" + str(AGENT_PARAMS.n_quantiles) + \
                      "\n     top_quantiles_to_drop_per_net=" + str(AGENT_PARAMS.top_quantiles_drop) + \
                      "\n     g_SDE=" + str(AGENT_PARAMS.gSDE) + \
                      "\n     seed=" + str(seed)

    return str_params_short, str_params_long


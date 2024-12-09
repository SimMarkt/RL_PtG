# Contains classes for processing the training of RL for dynamic Real-Time-Optimization of Power-to-Gas Processes

import numpy as np


class GlobalParams:
    def __init__(self):
        # economical scenario:  1 > Involves gas and emission market data,
        #                       2 > Involves a fixed gas price and emission market data,
        #                       3 > Involves EEG-procees for ch4 without emission market data (BHKW fired)
        self.scenario = 1
        self.operation = "OP1"  # specifies the load level "OP1" or "OP2"
        self.total_steps = 20000000         # total number of steps after training [588235, 1764705, 6470588, 20000000]
        self.initial_n_steps = 15000000      # initial number of steps in pre-training
        self.num_loops = 5          # Number of loops over the total training set
        self.train_len_d = 46496    # Total number of days in the training set
        self.price_ahead = 13       # Number of values for electricity price future data (0 - 12h)
        self.n_envs = 6           # Number environments/workers for training
        self.time_step_op = 2       # Time step between consecutive entries in the methanation operation data sets in sec
        self.noise = 10             # Noise when changing the methanation state in the gym env
                                    # [# of steps in operation data set]
        self.parallel = "Singleprocessing"      # specifies setup for "Singleprocessing" (DummyVecEnv)
                                                # or "Multiprocessing" (SubprocVecEnv)
        self.train_or_eval = "train"            # specifies whether the env info should be provided "eval"
                                                # or not "train" (recommended for training)
        self.action_type = "continuous"  # the type of action: "discrete" or "continuous" ##############################################################

        # data file paths:
        self.datafile_path_train_el = "../data/spot_market_data/data-day-ahead-el-train_separate_syn_aug6.csv"
        self.datafile_path_cv_el = "../data/spot_market_data/data-day-ahead-el-cv_separate.csv"
        self.datafile_path_test_el = "../data/spot_market_data/data-day-ahead-el-test_separate.csv"
        self.datafile_path_train_gas = "../data/spot_market_data/data-day-ahead-gas-train_separate_syn_aug6.csv"
        self.datafile_path_cv_gas = "../data/spot_market_data/data-day-ahead-gas-cv_separate.csv"
        self.datafile_path_test_gas = "../data/spot_market_data/data-day-ahead-gas-test_separate.csv"
        self.datafile_path_train_eua = "../data/spot_market_data/data-day-ahead-eua-train_separate_syn_aug6.csv"
        self.datafile_path_cv_eua = "../data/spot_market_data/data-day-ahead-eua-cv_separate.csv"
        self.datafile_path_test_eua = "../data/spot_market_data/data-day-ahead-eua-test_separate.csv"
        if self.operation == "OP1":
            self.datafile_path2 = "../data/OP1/data-meth_startup_cold.csv"
            self.datafile_path3 = "../data/OP1/data-meth_startup_hot.csv"
            self.datafile_path4 = "../data/OP1/data-meth_cooldown.csv"
            self.datafile_path5 = "../data/OP1/data-meth_standby_down.csv"  # from operation to Hot-Standby
            self.datafile_path6 = "../data/OP1/data-meth_standby_up.csv"  # from shutdown to Hot-Standby
            self.datafile_path7 = "../data/OP1/data-meth_op1_start_p.csv"
            self.datafile_path8 = "../data/OP1/data-meth_op2_start_f.csv"
            self.datafile_path9 = "../data/OP1/data-meth_op3_p_f.csv"
            self.datafile_path10 = "../data/OP1/data-meth_op4_p_f_p_5.csv"
            self.datafile_path11 = "../data/OP1/data-meth_op5_p_f_p_10.csv"
            self.datafile_path12 = "../data/OP1/data-meth_op6_p_f_p_15.csv"
            self.datafile_path13 = "../data/OP1/data-meth_op7_p_f_p_20.csv"
            self.datafile_path14 = "../data/OP1/data-meth_op8_f_p.csv"
            self.datafile_path15 = "../data/OP1/data-meth_op9_f_p_f_5.csv"
            self.datafile_path16 = "../data/OP1/data-meth_op10_f_p_f_10.csv"
            self.datafile_path17 = "../data/OP1/data-meth_op11_f_p_f_15.csv"
            self.datafile_path18 = "../data/OP1/data-meth_op12_f_p_f_20.csv"
        elif self.operation == "OP2":
            self.datafile_path2 = "../data/OP2/data-meth_startup_cold.csv"
            self.datafile_path3 = "../data/OP2/data-meth_startup_hot.csv"
            self.datafile_path4 = "../data/OP2/data-meth_cooldown.csv"
            self.datafile_path5 = "../data/OP2/data-meth_standby_down.csv"  # from operation to Hot-Standby
            self.datafile_path6 = "../data/OP2/data-meth_standby_up.csv"  # from shutdown to Hot-Standby
            self.datafile_path7 = "../data/OP2/data-meth_op1_start_p_12kW.csv"
            self.datafile_path8 = "../data/OP2/data-meth_op2_start_f_12kW.csv"
            self.datafile_path9 = "../data/OP2/data-meth_op3_p_f_12kW.csv"
            self.datafile_path10 = "../data/OP2/data-meth_op4_p_f_p_5_12kW.csv"
            self.datafile_path11 = "../data/OP2/data-meth_op5_p_f_p_10_12kW.csv"
            self.datafile_path12 = "../data/OP2/data-meth_op6_p_f_p_15_12kW.csv"
            self.datafile_path13 = "../data/OP2/data-meth_op7_p_f_p_20_12kW.csv"
            self.datafile_path14 = "../data/OP2/data-meth_op8_f_p_12kW.csv"
            self.datafile_path15 = "../data/OP2/data-meth_op9_f_p_f_5_12kW.csv"
            self.datafile_path16 = "../data/OP2/data-meth_op10_f_p_f_10_12kW.csv"
            self.datafile_path17 = "../data/OP2/data-meth_op11_f_p_f_15_12kW.csv"
            self.datafile_path18 = "../data/OP2/data-meth_op12_f_p_f_20_12kW.csv"

        # Control and inner state spaces of the Power-to-Gas System (aligned with the programmable logic controller)
        self.ptg_state_space = {
            'standby': 0,
            'cooldown': 1,
            'startup': 2,
            'partial_load': 3,
            'full_load': 4,
        }

        # methanation data for stationary operation
        if self.operation == "OP1":
            self.meth_stats_load = {  # [off, partial_load, full_load] = [0 %, 8.2 %, 23.2 %]
                'Meth_State': [2, 5, 5],  # addresses the inner state spaces of the Power-to-Gas System
                'Meth_Action': [6, 10, 11],  # addresses the action spaces of the Power-to-Gas System
                'Meth_Hot_Cold': [0, 1, 1],  # hot [=1] or cold [=0] methanation reactor
                'Meth_T_cat': [11.0, 451.0, 451.0],
                # maximum catalyst temperature in the methanation reactor system [°C]
                'Meth_H2_flow': [0.0, 0.00701, 0.0198],  # hydrogen reactant molar flow [mol/s]
                'Meth_CH4_flow': [0.0, 0.00172, 0.0048],  # methane product molar flow [mol/s]
                'Meth_H2_res_flow': [0.0, 0.000054, 0.000151],  # hydrogen product molar flow (residues) [mol/s]
                'Meth_H2O_flow': [0.0, 0.0624, 0.458545],  # water mass flow [kg/h]
                'Meth_el_heating': [0.0, 231.0, 350.0]
                # electrical power consumption for heating the methanation plant [W]
            }
        elif self.operation == "OP2":
            self.meth_stats_load = {  # [off, partial_load, full_load] = [0 %, 23.2 %, 56.7 %]
                'Meth_State': [2, 5, 5],  # addresses the inner state spaces of the Power-to-Gas System
                'Meth_Action': [6, 10, 11],  # addresses the action spaces of the Power-to-Gas System
                'Meth_Hot_Cold': [0, 1, 1],  # hot [=1] or cold [=0] methanation reactor
                'Meth_T_cat': [11.0, 451.0, 451.0],
                # maximum catalyst temperature in the methanation reactor system [°C]
                'Meth_H2_flow': [0.0, 0.0198, 0.0485],  # hydrogen reactant molar flow [mol/s]
                'Meth_CH4_flow': [0.0, 0.0048, 0.0114],  # methane product molar flow [mol/s]
                'Meth_H2_res_flow': [0.0, 0.000151, 0.0017],  # hydrogen product molar flow (residues) [mol/s]
                'Meth_H2O_flow': [0.0, 0.458545, 1.22],  # water mass flow [kg/h]
                'Meth_el_heating': [0.0, 350.0, 380.0]
                # electrical power consumption for heating the methanation plant [W]
            }
        else:
            assert False, 'Wrong Operation specified - ["OP1", "OP2"]'

        # price data
        self.ch4_price_fix = 15.0               # ct/kWh (incl. CH4 sale)
        self.heat_price = 4.6                   # ct/kWh
        self.o2_price = 10.2                    # ct/Nm³
        self.water_price = 6.4                  # ct/m³
        self.eeg_el_price = 17.84               # ct/kWh_el
        # species properties and efficiencies
        self.H_u_CH4 = 35.883                   # MJ/m³ (lower heating value)
        self.H_u_H2 = 10.783                    # MJ/m³ (lower heating value)
        self.h_H2O_evap = 2257                  # kJ/kg (at 1 bar)
        self.dt_water = 90                      # K (Tempature difference between cooling water and evaporation)
        self.cp_water = 4.18                    # kJ/kgK
        self.rho_water = 998                    # kg/m³
        # convert_mol_to_Nm3 = R_uni * T_0 / p_0 = 8.3145J/mol/K * 273.15K / 101325Pa = 0.02241407 Nm3/mol
        self.convert_mol_to_Nm3 = 0.02241407    # For an ideal gas at normal conditions
        self.Molar_mass_CO2 = 44.01             # g/mol molar mass of carbon dioxid
        self.Molar_mass_H2O = 18.02             # g/mol molar mass of water
        self.min_load_electrolyzer = 0.032      # = 3.2%  - minimum electrolyzer load
        self.max_h2_volumeflow = self.convert_mol_to_Nm3 * self.meth_stats_load['Meth_H2_flow'][2]  # m³/s - Experimental maximum electrolyzer power
        self.eta_BHKW = 0.38                    # BHKW efficiency

        self.r_0_values = {  # Reward level price values -> Sets general height of the Reward penalty
            'el_price': [0],
            'gas_price': [10],
            'eua_price': [50],
        }

        # threshold values for methanation data
        self.t_cat_standby = 188.2              # °C (catalyst temperature threshold for changing standby data set)
        self.t_cat_startup_cold = 160           # °C (catalyst temperature threshold for cold start conditions)
        self.t_cat_startup_hot = 350            # °C (catalyst temperature threshold for hot start conditions)
        # time threshold for load change data set, from time = 0
        self.time1_start_p_f = 1201             # simulation step -> 2400 sec
        self.time2_start_f_p = 151              # simulation step -> 300 sec
        self.time_p_f = 210                     # simulation steps for load change (asc) -> 420 sec
        self.time_f_p = 126                     # simulation steps for load change (des) -> 252 sec
        self.time1_p_f_p = 51                   # simulation step -> 100 sec
        self.time2_p_f_p = 151                  # simulation step -> 300 sec
        self.time23_p_f_p = 225                 # simulation step inbetween time2_p_f_p and time3_p_f_p
        self.time3_p_f_p = 301                  # simulation step -> 600 sec
        self.time34_p_f_p = 376                 # simulation step inbetween time3_p_f_p and time4_p_f_p
        self.time4_p_f_p = 451                  # simulation step -> 900 sec
        self.time45_p_f_p = 563                 # simulation step inbetween time4_p_f_p and time5_p_f_p
        self.time5_p_f_p = 675                  # simulation step -> 1348 sec
        self.time1_f_p_f = 51                   # simulation step -> 100 sec
        self.time2_f_p_f = 151                  # simulation step -> 300 sec
        self.time23_f_p_f = 225                 # simulation step inbetween time2_f_p_f and time3_f_p_f
        self.time3_f_p_f = 301                  # simulation step -> 600 sec
        self.time34_f_p_f = 376                 # simulation step inbetween time3_f_p_f and time4_f_p_f
        self.time4_f_p_f = 451                  # simulation step -> 900 sec
        self.time45_f_p_f = 526                 # simulation step inbetween time4_f_p_f and time5_f_p_f
        self.time5_f_p_f = 601                  # simulation step -> 1200 sec
        # simulation steps for fully developed partial / full load
        self.i_fully_developed = 12000          # simulation step -> 24000 sec (initial value)
        self.j_fully_developed = 100            # simulation step -> 24000 sec (step marker)

        # lower and upper bounds for gym observations
        self.el_l_b = -10
        self.el_u_b = 80
        self.gas_l_b = 0.4
        self.gas_u_b = 31.6
        self.eua_l_b = 23
        self.eua_u_b = 98
        self.T_l_b = 10
        self.T_u_b = 600
        self.h2_l_b = 0
        self.h2_u_b = self.meth_stats_load['Meth_H2_flow'][2]
        self.ch4_l_b = 0
        self.ch4_u_b = self.meth_stats_load['Meth_CH4_flow'][2]
        self.h2_res_l_b = 0
        self.h2_res_u_b = self.meth_stats_load['Meth_H2_res_flow'][2]
        self.h2o_l_b = 0
        self.h2o_u_b = self.meth_stats_load['Meth_H2O_flow'][2]
        self.heat_l_b = 0
        self.heat_u_b = 1800




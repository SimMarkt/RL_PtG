# ----------------------------------------------------------------------------------------------------------------
# RL_PtG: Deep Reinforcement Learning for Power-to-Gas dispatch optimization
# https://github.com/SimMarkt/RL_PtG

# rl_config_env: 
# > Contains the configuration of the environment (Power-to-Gas Process) 
# > Converts the config_env.yaml data into a class object for further processing
# ----------------------------------------------------------------------------------------------------------------

import yaml

class EnvConfiguration:
    def __init__(self):
        # Load the environment configuration
        with open("config/config_env.yaml", "r") as env_file:
            env_config = yaml.safe_load(env_file)
        
        self.scenario = env_config['scenario']                  # business case / economic scenario
        assert self.scenario in [1,2,3], f"Specified business scenario ({self.scenario}) must match one of the three implemented scenarios [1,2,3]!"
        self.operation = env_config['operation']                     # specifies the load level "OP1" or "OP2" of the PtG-CH4 plant
        self.train_len_d = None                                 # total number of days in the training set 
        self.price_ahead = env_config['price_ahead']            # number of forecast values for electricity price future data (0-12h)                    
        self.time_step_op = env_config['time_step_op']          # Time step between consecutive entries in the methanation operation data sets in sec
        self.noise = env_config['noise']                        # noise factor when changing the methanation state in the gym env [# of steps in operation data set]                         
        self.eps_len_d = env_config['eps_len_d']                # No. of days in an episode (episodes are randomly selected from the entire training data set without replacement)
        self.state_change_penalty = env_config['state_change_penalty'] # Factor which enables reward penalty during training (if state_change_penalty = 0.0: No reward penalty; if state_change_penalty > 0.0: Reward penalty on mode transitions;)
        self.sim_step = env_config['sim_step']        # Frequency for taking an action in [s] 

        # file paths of energy spot market data for training and evaluation:
        # (_train: training set; _val: validation set; _test: test set)
        # _el: electricity spot market data
        self.datafile_path_train_el = env_config['datafile_path_train_el']
        self.datafile_path_val_el = env_config['datafile_path_val_el']
        self.datafile_path_test_el = env_config['datafile_path_test_el']
        # _gas: natural gas/SNG spot market data
        self.datafile_path_train_gas = env_config['datafile_path_train_gas']
        self.datafile_path_val_gas = env_config['datafile_path_val_gas']
        self.datafile_path_test_gas = env_config['datafile_path_test_gas']
        # _eua: European emission allowances (EUA) spot market data
        self.datafile_path_train_eua = env_config['datafile_path_train_eua']
        self.datafile_path_val_eua = env_config['datafile_path_val_eua']
        self.datafile_path_test_eua = env_config['datafile_path_test_eua']

        # file paths of process data for the dynamic data-based process model of the methanation plant depending on the load level:
        assert self.operation in ['OP1', 'OP2'], f"Wrong load level specified - data/config_env.yaml -> operation : {env_config['operation']} must match ['OP1', 'OP2']"
        base_path = env_config['datafile_path']['path'] + self.operation
        for i in range(2, 19):  # For datafile_path2 to datafile_path18
            setattr(self, f'datafile_path{i}', f"{base_path}/{env_config['datafile_path']['datafile'][f'datafile_path{i}']}")

        self.ptg_state_space = env_config['ptg_state_space']                            # control and inner state spaces of the Power-to-Gas System (aligned with the programmable logic controller)
        self.meth_stats_load = env_config['meth_stats_load'][self.operation]                 # methanation data for steady-state operation for the different load levels
        self.r_0_values = env_config['r_0_values']                                      # Reward level price values -> Sets general height of the Reward penalty

        # economic data and parameters
        self.ch4_price_fix = env_config['ch4_price_fix']                    # fixed SNG price in business case/ scenario 2 in [ct/kWh]
        self.heat_price = env_config['heat_price']                          # heat price in [ct/kWh]
        self.o2_price = env_config['o2_price']                              # O2 price in [ct/Nm³]
        self.water_price = env_config['water_price']                        # water price in [ct/m³]
        self.eeg_el_price = env_config['eeg_el_price']                      # EEG tender price in business case/ scenario 3 in [ct/kWh_el]
        # species properties and efficiencies
        self.H_u_CH4 = env_config['H_u_CH4']                                # lower heating value of methane in [MJ/m³]
        self.H_u_H2 = env_config['H_u_H2']                                  # lower heating value of hydrogen in [MJ/m³]
        self.h_H2O_evap = env_config['h_H2O_evap']                          # specific enthalpy of vaporization in [kJ/kg] (at 0.1 MPa)
        self.dt_water = env_config['dt_water']                              # tempature difference between cooling water and evaporation in [K]
        self.cp_water = env_config['cp_water']                              # specific heat capacity of water in [kJ/kgK]
        self.rho_water = env_config['rho_water']                            # densitiy of water in [kg/m³]
        self.convert_mol_to_Nm3 = env_config['convert_mol_to_Nm3']          # factor for converting moles to Nm³ for an ideal gas at normal conditions in [Nm³/mol] (convert_mol_to_Nm3 : R_uni * T_0 / p_0 = 8.3145J/mol/K * 273.15K / 101325Pa = 0.02241407 Nm3/mol)
        self.Molar_mass_CO2 = env_config['Molar_mass_CO2']                  # molar mass of carbon dioxid in [g/mol]
        self.Molar_mass_H2O = env_config['Molar_mass_H2O']                  # molar mass of water in [g/mol]
        self.min_load_electrolyzer = env_config['min_load_electrolyzer']    # minimum electrolyzer load = 3.2% (According to static data-based regression model of PEMEL)
        self.eta_CHP = env_config['eta_CHP']                                # gas engine/ CHP efficiency
        self.max_h2_volumeflow = self.convert_mol_to_Nm3 *  self.meth_stats_load['Meth_H2_flow'][2]

        # threshold values for methanation data
        self.t_cat_standby = env_config['t_cat_standby']                # catalyst temperature threshold for changing standby data set in [°C] (If the plant goes to standby from idle state and reaches T_Cat > t_cat_standby, the model uses "data-meth_startup_hot.csv" for the next startup)
        self.t_cat_startup_cold = env_config['t_cat_startup_cold']      # catalyst temperature threshold for cold start conditions in [°C] (For T_Cat < t_cat_startup_cold, the model uses "data-meth_startup_cold.csv" for the next startup)
        self.t_cat_startup_hot = env_config['t_cat_startup_hot']        # catalyst temperature threshold for hot start conditions in [°C] (For T_Cat > t_cat_startup_hot, the model uses "data-meth_startup_hot.csv" for the next startup)
        # time threshold for load change data set, from time = 0 (Important to identify the data set with the most suitable mode transition)
        self.time1_start_p_f = env_config['time1_start_p_f']            # simulation step -> 2400 sec
        self.time2_start_f_p = env_config['time2_start_f_p']            # simulation step -> 300 sec
        self.time_p_f = env_config['time_p_f']                          # simulation steps for load change (asc) -> 420 sec
        self.time_f_p = env_config['time_f_p']                          # simulation steps for load change (des) -> 252 sec
        self.time1_p_f_p = env_config['time1_p_f_p']                    # simulation step -> 100 sec
        self.time2_p_f_p = env_config['time2_p_f_p']                    # simulation step -> 300 sec
        self.time23_p_f_p = env_config['time23_p_f_p']                  # simulation step inbetween time2_p_f_p and time3_p_f_p
        self.time3_p_f_p = env_config['time3_p_f_p']                    # simulation step -> 600 sec
        self.time34_p_f_p = env_config['time34_p_f_p']                  # simulation step inbetween time3_p_f_p and time4_p_f_p
        self.time4_p_f_p = env_config['time4_p_f_p']                    # simulation step -> 900 sec
        self.time45_p_f_p = env_config['time45_p_f_p']                  # simulation step inbetween time4_p_f_p and time5_p_f_p
        self.time5_p_f_p = env_config['time5_p_f_p']                    # simulation step -> 1348 sec
        self.time1_f_p_f = env_config['time1_f_p_f']                    # simulation step -> 100 sec
        self.time2_f_p_f = env_config['time2_f_p_f']                    # simulation step -> 300 sec
        self.time23_f_p_f = env_config['time23_f_p_f']                  # simulation step inbetween time2_f_p_f and time3_f_p_f
        self.time3_f_p_f = env_config['time3_f_p_f']                    # simulation step -> 600 sec
        self.time34_f_p_f = env_config['time34_f_p_f']                  # simulation step inbetween time3_f_p_f and time4_f_p_f
        self.time4_f_p_f = env_config['time4_f_p_f']                    # simulation step -> 900 sec
        self.time45_f_p_f = env_config['time45_f_p_f']                  # simulation step inbetween time4_f_p_f and time5_f_p_f
        self.time5_f_p_f = env_config['time5_f_p_f']                    # simulation step -> 1200 sec
        # simulation steps for fully developed partial / full load transition
        self.i_fully_developed = env_config['i_fully_developed']        # simulation step -> 24000 sec (initial value)
        self.j_fully_developed = env_config['j_fully_developed']        # simulation step -> 24000 sec (step marker)

        # Lower and upper bounds for gym observations for normalization of the state features to accelerate learning (Should be adjusted to the respected value range)
        self.el_l_b = env_config['el_l_b']                              # lower bound of electricity prices in [ct/kWh_el]
        self.el_u_b = env_config['el_u_b']                              # upper bound of electricity prices in [ct/kWh_el]
        self.gas_l_b = env_config['gas_l_b']                            # lower bound of (S)NG prices in [ct/kWh_th]
        self.gas_u_b = env_config['gas_u_b']                            # upper bound of (S)NG prices in [ct/kWh_th]
        self.eua_l_b = env_config['eua_l_b']                            # lower bound of EUA prices in [€/t_CO2]
        self.eua_u_b = env_config['eua_u_b']                            # upper bound of EUA prices in [€/t_CO2]
        self.T_l_b = env_config['T_l_b']                              # lower bound of catalyst temperatures T_CAT in [°C]
        self.T_u_b = env_config['T_u_b']                              # upper bound of catalyst temperatures T_CAT in [°C]
        self.h2_l_b = env_config['h2_l_b']                             # lower bound of hydrogen molar flow in [mol/s]
        self.h2_u_b = self.meth_stats_load['Meth_H2_flow'][2]           # upper bound of hydrogen molar flow in [mol/s]
        self.ch4_l_b = env_config['ch4_l_b']                            # lower bound of methane molar flow in [mol/s]
        self.ch4_u_b = self.meth_stats_load['Meth_CH4_flow'][2]         # upper bound of methane molar flow in [mol/s]
        self.h2_res_l_b = env_config['h2_res_l_b']                         # lower bound of residual product gas hydrogen molar flow in [mol/s]
        self.h2_res_u_b = self.meth_stats_load['Meth_H2_res_flow'][2]   # upper bound of residual product gas hydrogen molar flow in [mol/s]
        self.h2o_l_b = env_config['h2o_l_b']                            # lower bound of water mass flow in [kg/h]
        self.h2o_u_b = self.meth_stats_load['Meth_H2O_flow'][2]         # upper bound of water mass flow in [kg/h]
        # The upper bound of hydrogen (h2_u_b), methane (ch4_u_b), residual hydrogen (h2_res_u_b), and water (h2o_u_b) equal the full_load values of meth_stats_load of the chosen load level
        self.heat_l_b = env_config['heat_l_b']                           # lower bound of the power consumption of methanation in [W]
        self.heat_u_b = env_config['heat_u_b']                           # upper bound of the power consumption of methanation in [W]


        

       




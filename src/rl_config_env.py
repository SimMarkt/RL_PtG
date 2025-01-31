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

        # Unpack data
        self.__dict__.update(env_config)
        
        assert self.scenario in [1,2,3], f"Specified business scenario ({self.scenario}) must match one of the three implemented scenarios [1,2,3]!"
        self.train_len_d = None                                 # total number of days in the training set                    
        raw_mod_set = ['raw', 'mod']
        assert self.raw_modified in raw_mod_set, f"Wrong type of state design specified - data/config_train.yaml -> raw_mod : {env_config['raw_modified']} must match {raw_mod_set}"
        
        # file paths of process data for the dynamic data-based process model of the methanation plant depending on the load level:
        assert self.operation in ['OP1', 'OP2'], f"Wrong load level specified - data/config_env.yaml -> operation : {env_config['operation']} must match ['OP1', 'OP2']"
        base_path = self.datafile_path['path'] + self.operation
        for i in range(2, 19):  # For datafile_path2 to datafile_path18
            setattr(self, f'datafile_path{i}', f"{base_path}/{self.datafile_path['datafile'][f'datafile_path{i}']}")

        self.meth_stats_load = self.meth_stats_load[self.operation]         # methanation data for steady-state operation for the different load levels
        self.max_h2_volumeflow = self.convert_mol_to_Nm3 *  self.meth_stats_load['Meth_H2_flow'][2]     # Maximum hydrogen production in the water electrolysis

        # Lower and upper bounds for gym observations for normalization of the state features to accelerate learning (Should be adjusted to the respected value range)
        self.h2_u_b = self.meth_stats_load['Meth_H2_flow'][2]           # upper bound of hydrogen molar flow in [mol/s]
        self.ch4_u_b = self.meth_stats_load['Meth_CH4_flow'][2]         # upper bound of methane molar flow in [mol/s]
        self.h2_res_u_b = self.meth_stats_load['Meth_H2_res_flow'][2]   # upper bound of residual product gas hydrogen molar flow in [mol/s]
        self.h2o_u_b = self.meth_stats_load['Meth_H2O_flow'][2]         # upper bound of water mass flow in [kg/h]

        # Variable names for statistics, data storage, and evaluation
        self.stats_names = ['steps_stats', 'el_price_stats', 'gas_price_stats', 'eua_price_stats', 'Meth_State_stats',
                            'Meth_Action_stats', 'Meth_Hot_Cold_stats', 'Meth_T_cat_stats', 'Meth_H2_flow_stats',
                            'Meth_CH4_flow_stats', 'Meth_H2O_flow_stats', 'Meth_el_heating_stats', 'Meth_ch4_revenues_stats',
                            'Meth_steam_revenues_stats', 'Meth_o2_revenues_stats', 'Meth_eua_revenues_stats',
                            'Meth_chp_revenues_stats', 'Meth_elec_costs_heating_stats', 'Meth_elec_costs_electrolyzer_stats',
                            'Meth_water_costs_stats', 'Meth_reward_stats', 'Meth_cum_reward_stats', 'pot_reward_stats', 'part_full_stats']

                    
                    
        


        

       




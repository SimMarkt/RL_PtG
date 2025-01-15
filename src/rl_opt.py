# Benchmark: Optimal solution for PtG-operation ignoring dynamics and a rule-based controller

import numpy as np
from tqdm import tqdm
import math

from src.rl_config_env import EnvConfig


def calculate_optimum(el_price_data: np.array, gas_price_data: np.array, eua_price_data: np.array, data_name: str):
    """
        Computes the maximum possible revenues and runs the plant when the potential reward > 0 ignoring any dynamics
        :param el_price_data: Electricity market data
        :param gas_price_data: Gas market data
        :param eua_price_data: EUA market data
        :param data_name: For tqdm output and reward level specification  ########################################################
        :return stats_dict_opt: Dictionary with methanation status values
    """

    EnvConfig = EnvConfig()

    meth_stats = EnvConfig.meth_stats_load

    # Define stats for data storage
    stats_names = ['steps_stats', 'el_price_stats', 'gas_price_stats', 'eua_price_stats', 'Meth_State_stats',
                   'Meth_Action_stats', 'Meth_Hot_Cold_stats', 'Meth_T_cat_stats', 'Meth_H2_flow_stats',
                   'Meth_CH4_flow_stats', 'Meth_H2O_flow_stats', 'Meth_el_heating_stats', 'Meth_ch4_revenues_stats',
                   'Meth_steam_revenues_stats', 'Meth_o2_revenues_stats', 'Meth_eua_revenues_stats',
                   'Meth_chp_revenues_stats', 'Meth_elec_costs_heating_stats', 'Meth_elec_costs_electrolyzer_stats',
                   'Meth_water_costs_stats', 'Meth_reward_stats', 'Meth_cum_reward_stats', 'partial_full_b']
    stats_dict_opt = {}
    stats = np.zeros((len(el_price_data), len(stats_names)))

    # Include reward calculation from CHP in the case of business scenario 3
    if EnvConfig.scenario == 3:
        b_s3 = 1
    else:
        b_s3 = 0

    rew_l = [0,1]     # first entry of the list is dedicated to partial load, second to full load

    cum_rew = 0  # cumulative reward

    for t in range(len(el_price_data)):     ######################################################## tqdm?
        t_day = int(math.floor(t / 24))
        for l in range(2):  # Reward calculation for both partial load and full load

            # Gas proceeds (Scenario 1+2):          If Scenario == 3: self.gas_price_h[0] = 0
            ch4_volumeflow = meth_stats['Meth_CH4_flow'][l+1] * EnvConfig.convert_mol_to_Nm3
            h2_res_volumeflow = meth_stats['Meth_H2_res_flow'][l+1] * EnvConfig.convert_mol_to_Nm3  # in Nm³/s
            Q_ch4 = ch4_volumeflow * EnvConfig.H_u_CH4 * 1000  # in kW
            Q_h2_res = h2_res_volumeflow * EnvConfig.H_u_H2 * 1000
            ch4_revenues = (Q_ch4 + Q_h2_res) * gas_price_data[t_day]  # in ct/h

            power_chp = Q_ch4 * EnvConfig.eta_CHP * b_s3  # in kW
            Q_chp = Q_ch4 * (1 - EnvConfig.eta_CHP) * b_s3  # in kW
            chp_revenues = power_chp * EnvConfig.eeg_el_price  # in ct/h

            Q_steam = meth_stats['Meth_H2O_flow'][l+1] * (EnvConfig.dt_water * EnvConfig.cp_water +
                                                          EnvConfig.h_H2O_evap) / 3600  # in kW
            steam_revenues = (Q_steam + Q_chp) * EnvConfig.heat_price  # in ct/h

            h2_volumeflow = meth_stats['Meth_H2_flow'][l+1] * EnvConfig.convert_mol_to_Nm3  # in Nm³/s
            o2_volumeflow = 1 / 2 * h2_volumeflow * 3600  # in Nm³/h = Nm³/s * 3600 s/h
            o2_revenues = o2_volumeflow * EnvConfig.o2_price  # in ct/h

            Meth_CO2_mass_flow = meth_stats['Meth_CH4_flow'][l+1] * EnvConfig.Molar_mass_CO2 / 1000  # in kg/s
            eua_revenues = Meth_CO2_mass_flow / 1000 * 3600 * eua_price_data[t_day] * 100  # in ct/h = kg/s * t/1000kg * 3600 s/h * €/t * 100 ct/€

            # Linear regression model for LHV efficiency of an 6 MW electrolyzer (maximum reference power)
            elec_costs_heating = meth_stats['Meth_el_heating'][l+1] / 1000 * el_price_data[t]  # in ct/h
            load_elec = h2_volumeflow / EnvConfig.max_h2_volumeflow
            if load_elec < EnvConfig.min_load_electrolyzer:
                eta_electrolyzer = 0.02
            else:
                eta_electrolyzer = (0.598 - 0.325 * load_elec ** 2 + 0.218 * load_elec ** 3 +
                                    0.01 * load_elec ** (-1) - 1.68 * 10 ** (-3) * load_elec ** (-2) +
                                    2.51 * 10 ** (-5) * load_elec ** (-3))
            elec_costs_electrolyzer = h2_volumeflow * EnvConfig.H_u_H2 * 1000 / eta_electrolyzer * el_price_data[t]
            elec_costs = elec_costs_heating + elec_costs_electrolyzer

            # Costs for water consumption:
            water_elec = meth_stats['Meth_H2_flow'][l+1] * EnvConfig.Molar_mass_H2O / 1000 * 3600  # in kg/h (1 mol water is consumed for producing 1 mol H2)
            water_costs = (meth_stats['Meth_H2O_flow'][l+1] + water_elec) / EnvConfig.rho_water * \
                          EnvConfig.water_price  # in ct/h = kg/h / kg/m³ * ct/m³

            rew_l[l] = (ch4_revenues + chp_revenues + steam_revenues + eua_revenues +
                        o2_revenues - elec_costs - water_costs)  # in ct/h

            # if t == 1000:             ################################################################### DELETE #####################################
            #     print(data_name)
            #     print("Meth_H2_flow", meth_stats['Meth_H2_flow'][l], "Meth_CH4_flow", meth_stats['Meth_CH4_flow'][l], "Meth_H2_res_flow", meth_stats['Meth_H2_res_flow'][l], "Meth_H2O_flow", meth_stats['Meth_H2O_flow'][l],"Meth_el_heating", meth_stats['Meth_el_heating'][l],"el_price_h[0]",el_price_data[t], "gas_price_h[0]", gas_price_data[t_day], "eua_price_h[0]",eua_price_data[t_day])
            #     print("h2_volumeflow", h2_volumeflow, "ch4_volumeflow", ch4_volumeflow, "h2_res_volumeflow",
            #           h2_res_volumeflow, "Q_ch4", Q_ch4, "Q_h2_res",
            #           Q_h2_res, "Meth_CO2_mass_flow", Meth_CO2_mass_flow)
            #     print("ch4_revenues",ch4_revenues,"chp_revenues",chp_revenues,"steam_revenues",steam_revenues,"o2_revenues",o2_revenues,"eua_revenues",eua_revenues,"elec_costs_heating",elec_costs_heating,"elec_costs_electrolyzer",elec_costs_electrolyzer,"water_costs",water_costs )
            #     print(load_elec, eta_electrolyzer, l)
            #     print("REW:", (ch4_revenues + chp_revenues + steam_revenues + eua_revenues + o2_revenues - elec_costs - water_costs))

        tmp = max(rew_l)
        index = rew_l.index(tmp)

        rew = max(rew_l)
        # print("REW_max=", rew)

        stats[t, 0] = t                         # counts the simulated hours ##################MAKE SHORTER ###############################
        stats[t, 1] = el_price_data[t]
        stats[t, 2] = gas_price_data[t_day]
        stats[t, 3] = eua_price_data[t_day]

        if rew > 0:                     ##################MAKE SHORTER ###############################
            stats[t, 4] = meth_stats['Meth_State'][index + 1]
            stats[t, 5] = meth_stats['Meth_Action'][index + 1]
            stats[t, 6] = meth_stats['Meth_Hot_Cold'][index + 1]
            stats[t, 7] = meth_stats['Meth_T_cat'][index + 1]
            stats[t, 8] = meth_stats['Meth_H2_flow'][index + 1]
            stats[t, 9] = meth_stats['Meth_CH4_flow'][index + 1]
            stats[t, 10] = meth_stats['Meth_H2O_flow'][index + 1]
            stats[t, 11] = meth_stats['Meth_el_heating'][index + 1]
            stats[t, 12] = ch4_revenues
            stats[t, 13] = steam_revenues
            stats[t, 14] = o2_revenues
            stats[t, 15] = eua_revenues
            stats[t, 16] = chp_revenues
            stats[t, 17] = -elec_costs_heating
            stats[t, 18] = -elec_costs_electrolyzer
            stats[t, 19] = -water_costs
            cum_rew += rew                          # cum_rew in [ct] -> cum_rew = rew * 1h = [ct/h * 1h] = [ct]
            stats[t, 22] = index
        else:                       ##################MAKE SHORTER ###############################
            stats[t, 4] = meth_stats['Meth_State'][0]
            stats[t, 5] = meth_stats['Meth_Action'][0]
            stats[t, 6] = meth_stats['Meth_Hot_Cold'][0]
            stats[t, 7] = meth_stats['Meth_T_cat'][0]
            stats[t, 8] = meth_stats['Meth_H2_flow'][0]
            stats[t, 9] = meth_stats['Meth_CH4_flow'][0]
            stats[t, 10] = meth_stats['Meth_H2O_flow'][0]
            stats[t, 11] = meth_stats['Meth_el_heating'][0]
            stats[t, 12] = 0
            stats[t, 13] = 0
            stats[t, 14] = 0
            stats[t, 15] = 0
            stats[t, 16] = 0
            stats[t, 17] = 0
            stats[t, 18] = 0
            stats[t, 19] = 0
            stats[t, 22] = -1

        stats[t, 20] = rew
        stats[t, 21] = cum_rew

    for m in range(len(stats_names)):
        stats_dict_opt[stats_names[m]] = stats[:, m]

    if data_name != "reward_Level":
        max_pot_cum_rew = stats_dict_opt['Meth_cum_reward_stats'][-EnvConfig.price_ahead]
    else:
        max_pot_cum_rew = stats_dict_opt['Meth_cum_reward_stats'][0]

    # for i in range(400):
    #     print(str(stats_dict_opt['el_price_stats'][i]) + ";" + str(stats_dict_opt['Meth_reward_stats'][i]) + ";" + str(stats_dict_opt['partial_full_b'][i]))


    print("--- ", data_name, ": Cumulative reward - theoretical optimum T-OPT = ", max_pot_cum_rew)

    return stats_dict_opt
















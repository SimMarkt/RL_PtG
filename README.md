# RL_PtG

Deep Reinforcement Learning (RL) for dynamic Real-time optimization of Power-to-Gas (PtG) dispatch with respect to Day-ahead electricity, natural gas, and emission allowances market data. The PtG process comprises a proton exchange membrane electrolyzer (PEMEL) and a chemical methanation unit.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [License](#license)
5. [Citing](#citing)
6. [Acknowledgments](#acknowledgments)

---

## Overview

Deep RL is a promising approach for economic optimization of chemical plant operation. This python project implements deep RL for PtG dispatch optimization under Day-ahead energy
market conditions. The file "rl_main.py" contains the for training RL agents using a "data-based process model" of PtG as environment. This model has been derived from experimental data of a real PtG demonstration plant and serve as environment, along with energy market data.
The environment has been implemented using the Gymnasium environment.
With regard to RL, the project incorporates six state-of-the-art RL algorithms (DQN, A2C, PPO, TD3, SAC, TQC) from Stable-Baselines3 library.

To configure the code, the project provides two YAML files in "./config": config_agent.yaml (for the RL agents) and config_env.yaml (for the environment)

The experimental process data and energy market data are present in "./data".
Note that two different load levels are ...

![Screenshot](screenshot.png)

For more information on the data-based process model, please refer to ...

- Data: Electricity price day-ahead data from SMARD; Since the main study used gas and EUA market data provided by MONTEL without the rights to publish. Create synthesized data based on the real market data using TimeGAN algorithm.

---

## Project Structure

The project is organized into the following directories and files:

```plaintext
RL_PtG/
│
├── config/
│   ├── config_agent.yaml
│   ├── config_env.yaml
│   └── config_train.yaml
│
├── data/
│   ├── OP1/
│   ├── OP2/
│   └── spot_market_data/
│
├── logs/
│
├── plots/
│
├── src/
│   ├── rl_config_agent.py
│   ├── rl_config_env.py
│   ├── rl_config_train.py
│   ├── rl_opt.py
│   └── rl_utils.py
│
├── tensorboard/
│
├── requirements.txt
├── rl_main.py
└── rl_tb.py

```

### `config/`
Contains configuration files for the project:
- **`config/config_agent.yaml`**: Configuration for the RL agent.
- **`config/config_env.yaml`**: Configuration for PtG environment.
- **`config/config_train.yaml`**: Configuration for training procedure.

### `data/`
Contains process data for two different load levels OP1 and OP2 with different dynamics and energy market data:
- **`data/OP.../data-meth_cooldown.csv`**: C
- **`data/OP.../data-meth_op1_start_p.csv`**: C
- **`data/OP.../data-meth_op2_start_f.csv`**: C
- **`data/OP.../data-meth_op3_p_f.csv`**: C
- **`data/OP.../data-meth_op4_p_f_p_5.csv`**: C
- **`data/OP.../data-meth_op5_p_f_p_10.csv`**: C
- **`data/OP.../data-meth_op6_p_f_p_15.csv`**: C
- **`data/OP.../data-meth_op7_p_f_p_20.csv`**: C
- **`data/OP.../data-meth_op8_f_p.csv`**: C
- **`data/OP.../data-meth_op9_f_p_f_5.csv`**: C
- **`data/OP.../data-meth_op10_f_p_f_10.csv`**: C
- **`data/OP.../data-meth_op11_f_p_f_15.csv`**: C
- **`data/OP.../data-meth_op12_f_p_f_20.csv`**: C
- **`data/OP.../data-meth_standby_down.csv`**: C
- **`data/OP.../data-meth_standby_up.csv`**: C
- **`data/OP.../data-meth_startup_cold.csv`**: C
- **`data/OP.../data-meth_startup_hot.csv`**: C
- **`data/OP.../data-meth_cooldown.csv`**: C
- **`data/spot_market_data/data-day-ahead-el-test.csv`**: C
- **`data/spot_market_data/data-day-ahead-el-train.csv`**: C
- **`data/spot_market_data/data-day-ahead-el-val.csv`**: C
- **`data/spot_market_data/data-day-ahead-eua-test.csv`**: C
- **`data/spot_market_data/data-day-ahead-eua-train.csv`**: C
- **`data/spot_market_data/data-day-ahead-eua-val.csv`**: C
- **`data/spot_market_data/data-day-ahead-gas-test.csv`**: C
- **`data/spot_market_data/data-day-ahead-gas-train.csv`**: C
- **`data/spot_market_data/data-day-ahead-gas-val.csv`**: C

### `logs/`
During training, RL_PtG stores the algorithm and its parameters with the best performance in the validation environment in 'logs/'.

### `plots/`
After the training procedure, the best algorithm/ policy is evaluated on the test set and RL_PtG will create a diagram of its performance in 'plots/'.

### `src/`
Contains source code for pre- and postprocessing:
- **`src/rl_config_agent.py`**: C
  - `AgentConfiguration()`: Class for preprocessing the agent's configuration.
    - `set_model()`: Specifies and initializes the Stable-Baselines3 model for RL training.
    - `load_model()`: Loads a pretrained Stable-Baselines3 model for RL training.
    - `save_model()`: Saves the trained Stable-Baselines3 model and its replay buffer (if applicable).
    - `get_hyper()`: Displays the algorithm's hyperparameters and creates a string for file identification using `get_hyper()`. 
    - `hyp_print()`: Displays the value of a specific hyperparameter and adds it to the string identifier.
- **`src/rl_config_env.py`**: C
  - `EnvConfiguration()`: Class for preprocessing the environment's configuration.
- **`src/rl_config_train.py`**: C
  - `TrainConfiguration()`: Class for preprocessing the training configuration.
- **`src/rl_opt.py`**: Computes the potential rewards, the load identifiers, and the theoretical optimum T-OPT ignoring plant dynamics.
  - `calculate_optimum()`: Computes the theoretical maximum revenue for the Power-to-Gas process, assuming no operational constraints.          
- **`src/rl_utils.py`**: Contains utiliy and helper functions
  - `import_market_data()`: Imports day-ahead market price data.
  - `import_data()`: Imports experimental methanation process data.
  

### `tensorboard/`
During RL training, RL_PtG will store a tensorboard file for monitoring.

### Main Script
- **`rl_main.py`**: The main script for training the predefined RL agent on the PtG dispatch task.

### Miscellaneous
- **`rl_tb.py`**: Returns the URL of the tensorboard server for monitoring of RL training results.
- **`requirements.txt`**: Contains the required python libraries.


### `src/`
Contains source code for the different threads and connection wrappers using object-oriented programming:
- **`src/pci_modbus.py`**: Implements the Modbus connection with a class object providing:
  - `connect()`: Connects to the Modbus client
  - `is_connected()`: Tests the Modbus connection
  - `read_pemel_status()`: Reads and interprets the Modbus register containing the current state of PEMEL using `convert_bits()`
  - `read_pemel_process_values()`: Reads the PEMEL process values using `convert_process_values()`
  - `convert_bits()`: Converts the binary signal of the bit-wise PEMEL state representation into a one-hot encoded array
  - `convert_process_values()`: Converts the process values in the different registers to an array
  - `write_pemel_current()`: Writes the set point of the PEMEL electrical current to the respective Modbus register using `convert_h2_flow_to_current()`
  - `convert_h2_flow_to_current()`: Converts the hydrogen flow rate to the PEMEL's electrical current using `interpolate_h2_flow()`
  - `interpolate_h2_flow()`: Determines the electrical current based on the experimental values in `PEMEL_Current_H2Flowrate.txt`
- **`src/pci_opcua.py`**: Implements the OPC UA connection with a class object providing:
  - `connect()`: Connects to the OPC UA server
  - `is_connected()`: Tests the OPC UA connection
  - `read_node_values()`: Reads the values of multiple nodes using their NodeIDs
- **`src/pci_sql.py`**: Implements the SQL connection with a class object providing:
  - `connect()`: Connects to the SQL database
  - `is_connected()`: Tests the SQL connection
  - `insert_data()`: Inserts data into PostgreSQL database
- **`src/threads.py`**: Implements multi-threaded operations, including:
  - **PEMEL control thread** > `pemel_control()`: Manages PEMEL operations using Modbus and OPC UA using `el_control_func()`
  - **Data storage thread** > `data_storage()`: Handles data transfer between the OPC UA server, Modbus client, and SQL database using `data_trans_func()`
  - **Supervisor thread** > `supervisor()`: Monitors and attempts reconnection for disconnected services.

### Main Scripts
- **`pci_main.py`**: The primary script for running multi-threaded data transfer operations.
- **`pci_main_ws.py`**: A variation of the main script designed to set up a Windows service for data transfer.

### Miscellaneous
- **`PEMEL_Current_H2Flowrate.txt`**: Contains the PEMEL hydrogen production depending on the applied electrical current.
- **`PyComInt.log`**: Contains the log for debugging and monitoring, will be created when running the code.
- **`requirements.txt`**: Contains the required python libraries.

---

## Installation

Detailed steps to set up the project on a local environment:

```bash
# Clone the repository
git clone https://github.com/SimMarkt/RL_PtG.git

# Navigate to the project directory
cd RL_PtG

```

Afterwards, create a new Python virtual environment in the project folder and install the packages in the requirements.txt.
Note that Python 3.10 or a newer Version is required to run the code.
After installing all Python packages, the code can be run by using the rl_main_TQC_hpc.py file.

---

## License

This project is licensed under [MIT License](LICENSE).

---

## Citing

If you use RL_PtG in your research please use the following BibTeX entry:
```BibTeX
@misc{SimMarkRLPtG,
  author = {Markthaler, Simon},
  title = {RL_PtG: Deep Reinforcement Learning for Power-to-Gas dispatch optimization},
  year = {2024},
  url = {https://github.com/SimMarkt/RL_PtG}
}
```

---

## Acknowledgments

This project was funded by the German *Federal Ministry for Economic Affairs and Climate Action* within the **Power-to-Biogas**
project (Project ID: 03KB165).

---
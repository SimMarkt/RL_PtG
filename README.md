# RL_PtG

Deep Reinforcement Learning (RL) for dynamic Real-time optimization of Power-to-Gas (PtG) dispatch with respect to Day-ahead electricity, natural gas, and emission allowances market data. The PtG process comprises a proton exchange membrane electrolyzer (PEMEL) and a chemical methanation unit.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [License](#license)
4. [Citing](#citing)
5. [Acknowledgments](#acknowledgments)

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

---

## Features

Highlight the main features of the project:
- Feature 1
- Feature 2
- Feature 3

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
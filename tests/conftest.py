"""
---------------------------------------------------------------------------------------------
RL_PtG: Deep Reinforcement Learning for Power-to-Gas Dispatch Optimization
GitHub Repository: https://github.com/SimMarkt/RL_PtG

conftest:
> Pytest fixtures for unit tests.
---------------------------------------------------------------------------------------------
"""

import pytest
import yaml

@pytest.fixture
def dummy_agent_config(tmp_path):
    """Provides a dummy agent configuration file."""
    config = {
        "rl_alg": "DQN",
        "hyperparameters": {
            "DQN": {
                "alpha": 0.001,
                "gamma": 0.99,
                "eps_init": 1.0,
                "eps_fin": 0.01,
                "eps_fra": 0.1,
                "buffer_size": 100000,
                "batch_size": 64,
                "hidden_layers": 3,
                "hidden_units": 64,
                "activation": "ReLU",
                "learning_starts": 5000,
                "tau": 0.005,
                "train_freq": 4,
                "target_update_interval": 1000,
            }
        }
    }
    config_path = tmp_path / "config_agent.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return str(config_path)

@pytest.fixture
def dummy_train_config(tmp_path):
    """Provides a dummy training configuration file."""
    config = {
        "com_conf": "pc",
        "model_conf": "simple_train",
        "r_seed_train": [89],
        "r_seed_test": [531]
    }
    config_path = tmp_path / "config_train.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return str(config_path)

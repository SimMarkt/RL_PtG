"""
---------------------------------------------------------------------------------------------
RL_PtG: Deep Reinforcement Learning for Power-to-Gas Dispatch Optimization
GitHub Repository: https://github.com/SimMarkt/RL_PtG

test_config_agent:
> Test cases for the AgentConfiguration class in rl_config_agent.py.
---------------------------------------------------------------------------------------------
"""

# pylint: disable=no-member

import builtins

import pytest
import yaml

from src.rl_config_agent import AgentConfiguration

real_open = builtins.open

def test_agent_config_load(monkeypatch, dummy_agent_config):
    """Test loading of agent configuration."""
    def open_patch(file, *args, **kwargs):
        if file == "config/config_agent.yaml":
            return real_open(dummy_agent_config, *args, **kwargs)
        return real_open(file, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", open_patch)
    agent = AgentConfiguration()
    assert agent.rl_alg == "DQN"
    assert "DQN" in agent.hyperparameters
    assert agent.hyperparameters["DQN"]["alpha"] == 0.001
    assert agent.hyperparameters["DQN"]["activation"] == "ReLU"

def test_agent_config_invalid_alg(monkeypatch, dummy_agent_config):
    """Test handling of invalid RL algorithm in configuration."""
    with real_open(dummy_agent_config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["rl_alg"] = "INVALID"
    def open_patch(file, *args, **kwargs):
        if file == "config/config_agent.yaml":
            return real_open(dummy_agent_config, *args, **kwargs)
        return real_open(file, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", open_patch)
    monkeypatch.setattr("yaml.safe_load", lambda f: config)
    with pytest.raises(Exception):
        AgentConfiguration()

import pytest
from unittest.mock import MagicMock, patch

from open_ctf.training.online_rl import VerlRolloutEnvironment, start_online_rl

def test_verl_rollout_env_init():
    """Test standard initialization and generalized kwargs of VerlRolloutEnvironment."""
    env = VerlRolloutEnvironment(
        platform="custom_platform",
        target="TEST-001",
        max_steps=10,
        custom_reward_weight=0.5,
        explore_threshold=0.1
    )
    
    assert env.platform == "custom_platform"
    assert env.target == "TEST-001"
    assert env.max_steps == 10
    assert env.env_kwargs.get("custom_reward_weight") == 0.5
    assert env.env_kwargs.get("explore_threshold") == 0.1
    assert env.env is None
    assert env.executor is None

@patch('open_ctf.training.online_rl.OpenCTFEnv')
def test_verl_rollout_env_lazy_init(mock_open_ctf_env):
    """Test that lazy initialization correctly sets up the OpenCTFEnv mock without execution."""
    env = VerlRolloutEnvironment(platform="xbow", target="XBEN-003")
    
    # Init OpenCTFEnv manually
    env._lazy_init()
    
    # Verify OpenCTFEnv was instantiated
    assert env.env is not None
    mock_open_ctf_env.assert_called_once()

def test_start_online_rl():
    """Test dynamic algorithm and model config passing."""
    # This shouldn't crash, it should just log safely due to Scaffold implementation
    algo_config = {"lr": 1e-4, "kl_penalty": 0.01}
    model_config = {"model_path": "path/to/latest_model", "quant": "4-bit"}
    env_config = {"platform": "local", "target": "box-01"}
    
    start_online_rl(
        algo="ppo",
        algo_config=algo_config,
        model_config=model_config,
        env_config=env_config
    )

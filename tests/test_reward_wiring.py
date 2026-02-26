"""Reward wiring tests for OpenCTFTextEnv."""

from open_ctf.envs.skyrl.openctf_env import OpenCTFTextEnv


def _extras() -> dict:
    return {
        "ground_truth_flag": "FLAG{test}",
        "optimal_steps": 5,
        "max_turns": 3,
    }


def test_default_reward_is_created_when_config_missing():
    env = OpenCTFTextEnv(extras=_extras())
    try:
        assert env._reward_fn is not None
    finally:
        env.close()


def test_default_reward_is_created_for_empty_reward_config():
    env = OpenCTFTextEnv(extras=_extras(), reward_config={})
    try:
        assert env._reward_fn is not None
    finally:
        env.close()


def test_invalid_reward_config_type_uses_defaults(caplog):
    caplog.set_level("WARNING")
    env = OpenCTFTextEnv(extras=_extras(), reward_config="invalid")
    try:
        assert env._reward_fn is not None
        assert any("Invalid reward_config type" in msg for msg in caplog.messages)
    finally:
        env.close()

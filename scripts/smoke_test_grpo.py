"""Smoke test: GRPO data conversion + SkyRL config build + env registration"""
import json
import os
import sys

print("=" * 60)
print("GRPO PIPELINE SMOKE TEST")
print("=" * 60)

# 1. Test data conversion
print("\n1. Data conversion...")
from open_ctf.training.grpo import _convert_grpo_data, _build_skyrl_config
from open_ctf.challenges.registry import ChallengeRegistry

reg = ChallengeRegistry("/workspace/open-ctf-env/configs/challenges/cybench.yaml")
result = _convert_grpo_data(
    "/workspace/open-ctf-env/data/grpo_cybench40.jsonl",
    "/workspace/open-ctf-env/outputs/grpo_smoke_test",
    registry=reg,
)
print(f"   Converted: {result}")

# Count samples
import jsonlines

samples = list(jsonlines.open(result))
print(f"   Samples: {len(samples)}")
targets = sum(1 for s in samples if s.get("target"))
print(f"   With targets: {targets}")

# 2. Test SkyRL config build
print("\n2. SkyRL config build...")
config = {
    "model": {"max_seq_length": 4096},
    "lora": {"r": 64, "alpha": 128},
    "grpo": {
        "batch_size": 1,
        "num_generations": 2,
        "max_completion_length": 1024,
        "epochs": 1,
    },
    "output": {"save_steps": 10},
}
skyrl_cfg = _build_skyrl_config(
    "/workspace/open-ctf-env/outputs/sft-nanbeige3b-merged",
    "/workspace/open-ctf-env/outputs/grpo_smoke_test",
    config,
    result,
)
print(f"   Strategy: {skyrl_cfg['trainer']['strategy']}")
print(f"   Generator backend: {skyrl_cfg['generator']['backend']}")
print(f"   n_samples_per_prompt: {skyrl_cfg['generator']['n_samples_per_prompt']}")
print(f"   env_class: {skyrl_cfg['environment']['env_class']}")

# 3. Test env registration + init
print("\n3. Environment registration + init...")
from skyrl_gym.envs import register
from open_ctf.envs.skyrl.openctf_env import OpenCTFTextEnv

register(
    id="openctf",
    entry_point=OpenCTFTextEnv,
    kwargs={"reward_config": {}},
)
print("   Registration: OK")

# Create env with a sample
sample = samples[0]
extras = {
    "ground_truth_flag": sample.get("ground_truth_flag"),
    "optimal_steps": sample.get("optimal_steps"),
    "target": sample.get("target"),
    "challenge_id": sample.get("challenge_id"),
}
env = OpenCTFTextEnv(extras=extras)
print(f"   Env created for: {extras['challenge_id']}")
print(f"   Target: {extras['target']}")
print(f"   Tools available: {len(env.tools)}")

# 4. Test env init with prompt
print("\n4. Env init with sample prompt...")
prompt = sample["prompt"]
init_result = env.init(prompt)
print(f"   Init returned prompt with {len(init_result[0])} messages")
sys_content = str(init_result[0][0].get("content", ""))
has_tools = "tool_call" in sys_content or "shell_command" in sys_content
print(f"   System message includes tools: {has_tools}")

# 5. Test tool call parsing
print("\n5. Tool call parsing...")
from open_ctf.envs.skyrl.openctf_env import parse_tool_calls

test_text = '<tool_call>{"name": "shell_command", "arguments": {"command": "echo hello"}}</tool_call>'
parsed = parse_tool_calls(test_text)
print(f"   Hermes format: {parsed}")
assert len(parsed) == 1 and parsed[0]["name"] == "shell_command"

# 6. Test step with a simulated tool call
print("\n6. Env step with shell_command...")
step_result = env.step(test_text)
obs = step_result.get("observations", [])
reward = step_result.get("reward", 0.0)
done = step_result.get("done", False)
print(f"   Observations: {len(obs)} messages")
print(f"   Reward: {reward}")
print(f"   Done: {done}")
if obs:
    obs_content = obs[0].get("content", "")[:200]
    print(f"   Output: {obs_content}")

# 7. Test reward function
print("\n7. Reward function...")
from open_ctf.rewards.reward import CTFReward

reward_fn = CTFReward()
print(f"   CTFReward: OK (callable={callable(reward_fn)})")

print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)

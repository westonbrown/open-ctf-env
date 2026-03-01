#!/usr/bin/env python3
"""Quick smoke test: run 1 synthetic generation episode and dump the output."""

import json
import logging
import os
import sys

# Ensure project is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from open_ctf.synthetic_data_generation.manifest import WorldManifest
from open_ctf.synthetic_data_generation.executor import SimulatedEnvironmentExecutor
from open_ctf.synthetic_data_generation.generator import LiteLLMAgentAdapter, SyntheticGenerator

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_synth_gen")

# --- Config ---
MODEL = "openrouter/stepfun/step-3.5-flash:free"
MANIFEST_PATH = "configs/synthetic_data_generation/default.yaml"
MAX_TURNS = 15

def main():
    logger.info("Loading manifest from %s", MANIFEST_PATH)
    manifest = WorldManifest.from_yaml(MANIFEST_PATH)

    logger.info("Manifest: name=%s, flag=%s", manifest.name, manifest.ground_truth_flag)
    logger.info("Hosts: %s", list(set(h.hostname for h in manifest.hosts.values())))
    logger.info("Files: %s", list(manifest.files.keys()))
    logger.info("Tool response patterns: %s", {
        tool: list(patterns.keys()) for tool, patterns in manifest.tool_responses.items()
    })

    # Run one episode
    logger.info("=" * 60)
    logger.info("Running episode with model=%s, max_turns=%d", MODEL, MAX_TURNS)
    logger.info("=" * 60)

    adapter = LiteLLMAgentAdapter(model_name=MODEL)
    executor = SimulatedEnvironmentExecutor(manifest=manifest, max_steps=MAX_TURNS)

    trace = adapter.run_episode(executor, executor._current_manifest, MAX_TURNS)

    # --- Dump results ---
    print("\n" + "=" * 60)
    print("TRACE RESULTS")
    print("=" * 60)

    messages = trace["messages"]
    print(f"\nTotal messages: {len(messages)}")
    print(f"Success: {trace['metadata']['success']}")
    print(f"Total turns: {trace['metadata']['total_turns']}")
    print(f"Ground truth flag: {trace['ground_truth_flag']}")

    print("\n--- MESSAGE FLOW ---")
    for i, m in enumerate(messages):
        role = m.get("role", "?")
        content = m.get("content", "") or ""

        if role == "system":
            print(f"\n[{i}] SYSTEM ({len(content)} chars):")
            print(f"    {content[:200]}...")
        elif role == "user":
            print(f"\n[{i}] USER ({len(content)} chars):")
            print(f"    {content[:200]}")
        elif role == "assistant":
            tool_calls = m.get("tool_calls", [])
            tc_names = [tc.get("function", {}).get("name", "?") if isinstance(tc, dict)
                       else tc.function.name for tc in tool_calls] if tool_calls else []
            print(f"\n[{i}] ASSISTANT (content={len(content)} chars, tool_calls={tc_names}):")
            if content:
                print(f"    Content: {content[:300]}")
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    name = fn.get("name", "?")
                    args = fn.get("arguments", "")
                else:
                    name = tc.function.name
                    args = tc.function.arguments
                print(f"    -> {name}({args[:150]})")
        elif role == "tool":
            name = m.get("name", "?")
            tc_id = m.get("tool_call_id", "?")
            print(f"\n[{i}] TOOL ({name}, id={tc_id[:20]}...):")
            print(f"    {content[:300]}")

    # Save full trace
    out_path = "data/test_synth_output.jsonl"
    os.makedirs("data", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(json.dumps(trace, default=str) + "\n")
    print(f"\n\nFull trace saved to {out_path}")

    # --- Compare vs sft_v6 format ---
    print("\n" + "=" * 60)
    print("FORMAT COMPARISON vs sft_v6.jsonl")
    print("=" * 60)

    sft_path = "data/sft_v6.jsonl"
    if os.path.exists(sft_path):
        with open(sft_path) as f:
            real = json.loads(f.readline())

        print(f"\n{'Field':<25} {'Real (sft_v6)':<35} {'Synthetic':<35}")
        print("-" * 95)

        # Top-level keys
        real_keys = sorted(real.keys())
        synth_keys = sorted(trace.keys())
        print(f"{'top-level keys':<25} {str(real_keys):<35} {str(synth_keys):<35}")

        # Message count
        print(f"{'message count':<25} {len(real['messages']):<35} {len(messages):<35}")

        # System prompt length
        real_sys = next((m for m in real['messages'] if m['role'] == 'system'), {})
        synth_sys = next((m for m in messages if m['role'] == 'system'), {})
        print(f"{'system prompt len':<25} {len(real_sys.get('content','')):<35} {len(synth_sys.get('content','')):<35}")

        # User message length
        real_user = next((m for m in real['messages'] if m['role'] == 'user'), {})
        synth_user = next((m for m in messages if m['role'] == 'user'), {})
        print(f"{'user msg len':<25} {len(real_user.get('content','')):<35} {len(synth_user.get('content','')):<35}")

        # Tool calls
        real_tcs = [tc['function']['name'] for m in real['messages'] for tc in m.get('tool_calls', [])]
        synth_tcs = []
        for m in messages:
            for tc in m.get('tool_calls', []):
                if isinstance(tc, dict):
                    synth_tcs.append(tc.get('function',{}).get('name','?'))
                else:
                    synth_tcs.append(tc.function.name)
        print(f"{'tool call count':<25} {len(real_tcs):<35} {len(synth_tcs):<35}")
        print(f"{'unique tools':<25} {str(set(real_tcs)):<35} {str(set(synth_tcs)):<35}")

        # Has <think> blocks
        real_think = any('<think>' in (m.get('content','') or '') for m in real['messages'] if m.get('role')=='assistant')
        synth_think = any('<think>' in (m.get('content','') or '') for m in messages if m.get('role')=='assistant')
        print(f"{'has <think> blocks':<25} {str(real_think):<35} {str(synth_think):<35}")

        # Metadata
        print(f"{'metadata.source':<25} {real['metadata'].get('source',''):<35} {trace['metadata'].get('source',''):<35}")
        print(f"{'metadata.platform':<25} {real['metadata'].get('platform',''):<35} {trace['metadata'].get('platform',''):<35}")

        # Flag format
        print(f"{'ground_truth_flag':<25} {real.get('ground_truth_flag','')[:30]:<35} {trace.get('ground_truth_flag','')[:30]:<35}")

        # Ends with flag_found?
        real_last_tool = real['messages'][-1].get('name','')
        synth_last = messages[-1] if messages else {}
        synth_last_role = synth_last.get('role', '')
        synth_last_name = synth_last.get('name', '') if synth_last_role == 'tool' else 'N/A'
        print(f"{'last msg':<25} {'tool:'+real_last_tool:<35} {synth_last_role+':'+synth_last_name:<35}")

if __name__ == "__main__":
    main()

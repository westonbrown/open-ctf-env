#!/usr/bin/env python3
"""Local test of OpenEnv integration (no GPU needed).

Tests:
1. Environment lifecycle (reset/step/state)
2. All tool types (shell_command, python_code, submit_flag)
3. HTTP server endpoints
4. Tool wrapper functions (for TRL tools= parameter)
5. Reward computation
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from open_ctf.envs.openenv import ToolAction, AgentEnv
from open_ctf.envs.openenv.server import AgentEnvironment, create_app
from open_ctf.training.tools import init_env, reset_env, close_env, shell_command, python_code, submit_flag


def test_direct_environment():
    """Test environment without HTTP."""
    print("\n" + "=" * 60)
    print("TEST 1: Direct Environment (no HTTP)")
    print("=" * 60)

    env = AgentEnvironment(
        target="http://test-target:8080",
        ground_truth="FLAG{direct_test}",
        max_steps=5,
    )

    # Reset
    obs = env.reset()
    assert not obs.done
    assert obs.reward == 0.0
    print(f"✓ Reset: {obs.stdout[:50]}")

    # Shell command
    action = ToolAction(tool_name="shell_command", arguments={"command": "echo test"})
    obs = env.step(action)
    assert "test" in obs.stdout
    assert obs.exit_code == 0
    assert obs.reward == 0.05
    print(f"✓ Shell: stdout={obs.stdout.strip()!r}, reward={obs.reward}")

    # Python code
    action = ToolAction(tool_name="python_code", arguments={"code": "print(1+1)"})
    obs = env.step(action)
    assert "2" in obs.stdout
    assert obs.reward == 0.05
    print(f"✓ Python: stdout={obs.stdout.strip()!r}, reward={obs.reward}")

    # Wrong flag
    action = ToolAction(tool_name="submit_flag", arguments={"content": "FLAG{wrong}"})
    obs = env.step(action)
    assert obs.done
    assert obs.reward == -0.2
    assert not obs.flag_correct
    print(f"✓ Wrong flag: done={obs.done}, reward={obs.reward}")

    # Reset and correct flag
    env.reset()
    action = ToolAction(tool_name="submit_flag", arguments={"content": "FLAG{direct_test}"})
    obs = env.step(action)
    assert obs.done
    assert obs.reward == 1.0
    assert obs.flag_correct
    assert obs.flag_submitted
    print(f"✓ Correct flag: reward={obs.reward}, flag_correct={obs.flag_correct}")

    print("\n✓ All direct environment tests passed")


def test_http_server():
    """Test HTTP server and client."""
    print("\n" + "=" * 60)
    print("TEST 2: HTTP Server + Client")
    print("=" * 60)

    # Start server in subprocess
    env_process = subprocess.Popen(
        ["python3", "-m", "uvicorn", "open_ctf.envs.openenv.server:app",
         "--host", "127.0.0.1", "--port", "8101", "--log-level", "warning"],
        env={**os.environ, "PYTHONPATH": "src", "GROUND_TRUTH": "FLAG{http_test}"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    time.sleep(3)  # Wait for server startup

    try:
        import requests
        # Wait for health check
        for i in range(10):
            try:
                resp = requests.get("http://127.0.0.1:8101/health", timeout=1)
                if resp.ok:
                    break
            except:
                time.sleep(0.5)
        else:
            raise RuntimeError("Server failed to become healthy")

        # Test with HTTP client
        client = AgentEnv(base_url="http://127.0.0.1:8101")

        # Reset
        result = client.reset()
        assert result.observation.stdout
        assert result.reward == 0.0
        print(f"✓ HTTP Reset: {result.observation.stdout[:50]}")

        # Shell
        result = client.step(ToolAction(
            tool_name="shell_command",
            arguments={"command": "uname -s"}
        ))
        assert result.observation.exit_code == 0
        print(f"✓ HTTP Shell: {result.observation.stdout.strip()}")

        # Python
        result = client.step(ToolAction(
            tool_name="python_code",
            arguments={"code": "import sys; print(sys.version_info.major)"}
        ))
        assert "3" in result.observation.stdout
        print(f"✓ HTTP Python: stdout={result.observation.stdout.strip()}")

        # State
        state = client.state()
        assert state.step_count == 2
        print(f"✓ HTTP State: step_count={state.step_count}, tools_used={state.tools_used}")

        # Correct flag
        client.reset()
        result = client.step(ToolAction(
            tool_name="submit_flag",
            arguments={"content": "FLAG{http_test}"}
        ))
        assert result.done
        assert result.reward == 1.0
        assert result.observation.flag_correct
        print(f"✓ HTTP Flag: reward={result.reward}, done={result.done}")

        print("\n✓ All HTTP tests passed")

    finally:
        env_process.terminate()
        env_process.wait(timeout=5)


def test_trl_tool_wrappers():
    """Test TRL tool wrapper functions."""
    print("\n" + "=" * 60)
    print("TEST 3: TRL Tool Wrappers (tools= parameter)")
    print("=" * 60)

    # Start server
    env_process = subprocess.Popen(
        ["python3", "-m", "uvicorn", "open_ctf.envs.openenv.server:app",
         "--host", "127.0.0.1", "--port", "8102", "--log-level", "warning"],
        env={**os.environ, "PYTHONPATH": "src", "GROUND_TRUTH": "FLAG{wrapper_test}"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    time.sleep(3)

    try:
        import requests
        # Wait for health check
        for i in range(10):
            try:
                resp = requests.get("http://127.0.0.1:8102/health", timeout=1)
                if resp.ok:
                    break
            except:
                time.sleep(0.5)
        else:
            raise RuntimeError("Server failed to become healthy")

        # Initialize tool wrappers
        init_env("http://127.0.0.1:8102")
        reset_env()

        # Test shell_command wrapper
        output = shell_command("echo 'TRL wrapper test'")
        assert "TRL wrapper test" in output
        print(f"✓ shell_command: {output.strip()}")

        # Test python_code wrapper
        output = python_code("result = 5 * 5\nprint(f'Result: {result}')")
        assert "25" in output
        print(f"✓ python_code: {output.strip()}")

        # Test submit_flag wrapper (wrong)
        output = submit_flag("FLAG{wrong}")
        assert "Incorrect" in output
        print(f"✓ submit_flag (wrong): {output[:50]}")

        # Reset and correct flag
        reset_env()
        output = submit_flag("FLAG{wrapper_test}")
        assert "Correct" in output
        print(f"✓ submit_flag (correct): {output[:50]}")

        close_env()
        print("\n✓ All TRL wrapper tests passed")

    finally:
        env_process.terminate()
        env_process.wait(timeout=5)


def main():
    print("\n" + "=" * 60)
    print("OpenEnv Integration Test Suite")
    print("=" * 60)

    try:
        test_direct_environment()
        test_http_server()
        test_trl_tool_wrappers()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nOpenEnv integration is ready for DGX Spark testing.")
        return 0

    except AssertionError as e:
        print(f"\n❌ Test assertion failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

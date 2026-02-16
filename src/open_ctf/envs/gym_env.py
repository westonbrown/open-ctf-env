"""Gymnasium environment for CTF challenges.

This wraps CTF challenges as a standard Gymnasium env for RL training.
Command execution is delegated to an executor (e.g. BoxPwnr DockerExecutor)
rather than using the old DockerManager directly.
"""

import re
import logging
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, Callable

logger = logging.getLogger(__name__)


class OpenCTFEnv(gym.Env):
    """Gymnasium environment that wraps CTF challenges for reinforcement learning.

    Key components:
        1. Action space: Text commands up to 1000 characters (shell commands).
        2. Observation space: stdout, return_code, and flag_captured status.
        3. reset(): Calls the provided reset_fn to prepare a fresh challenge.
        4. step(): Executes the agent's command via exec_fn, checks for flags.
        5. Episode limit: max_steps (default 50) to prevent infinite loops.

    The environment is executor-agnostic. Provide ``exec_fn`` and
    ``reset_fn`` callables to integrate with any backend (BoxPwnr
    DockerExecutor, SSH, local shell, etc.).
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        challenge_id: str,
        exec_fn: Optional[Callable[[str], Tuple[str, int]]] = None,
        reset_fn: Optional[Callable[[], None]] = None,
        cleanup_fn: Optional[Callable[[], None]] = None,
        max_steps: int = 50,
        flag_pattern: str = r"(FLAG|XBOW)\{.*?\}",
    ):
        """Initialize the CTF Gymnasium environment.

        Args:
            challenge_id: Identifier for the challenge.
            exec_fn: Callable(command) -> (stdout, return_code).
            reset_fn: Callable() to reset/restart the challenge.
            cleanup_fn: Callable() to clean up resources on close.
            max_steps: Maximum steps per episode.
            flag_pattern: Regex pattern for flag detection.
        """
        self.challenge_id = challenge_id
        self.exec_fn = exec_fn
        self.reset_fn = reset_fn
        self.cleanup_fn = cleanup_fn
        self.max_steps = max_steps
        self.flag_pattern = re.compile(flag_pattern)

        # Action Space: Text command (Shell)
        self.action_space = spaces.Text(max_length=1000)

        # Observation Space
        self.observation_space = spaces.Dict({
            "stdout": spaces.Text(max_length=10000),
            "return_code": spaces.Discrete(256),
            "flag_captured": spaces.Discrete(2),
        })

        self.current_step = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0

        if self.reset_fn is not None:
            try:
                self.reset_fn()
            except Exception as e:
                logger.error(f"Failed to reset environment: {e}")
                raise

        obs = {
            "stdout": "Environment initialized. Ready for commands.",
            "return_code": 0,
            "flag_captured": 0,
        }
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute a shell command and return the observation."""
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        terminated = False
        reward = 0.0

        if self.exec_fn is None:
            raise RuntimeError(
                "No exec_fn provided. Set exec_fn to execute commands."
            )

        stdout, return_code = self.exec_fn(action)

        # Check for flag
        flag_match = self.flag_pattern.search(stdout)
        if flag_match:
            reward = 1.0
            terminated = True
            logger.info(f"Flag captured! {flag_match.group(0)}")

        obs = {
            "stdout": stdout,
            "return_code": return_code,
            "flag_captured": 1 if reward > 0 else 0,
        }

        return obs, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        if self.cleanup_fn is not None:
            self.cleanup_fn()

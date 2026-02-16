"""OpenEnv-style environment interface (stub).

Placeholder for a server-based CTF environment API.
Can be implemented using BoxPwnr's DockerExecutor when needed.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class OpenCTFEnvironment:
    """Stub for a CTF environment server.

    Preserves the interface contract for future implementation
    using BoxPwnr executors.
    """

    def __init__(
        self,
        challenge_id: str = "sqli-login-1",
        max_steps: int = 50,
    ):
        self.challenge_id = challenge_id
        self.max_steps = max_steps
        self.step_count = 0

    def reset(self):
        """Reset the environment. Must be implemented with an executor."""
        self.step_count = 0
        logger.warning(
            "OpenCTFEnvironment.reset() is a stub. "
            "Re-implement with BoxPwnr DockerExecutor for full functionality."
        )
        return {
            "stdout": "Environment stub - not connected to an executor.",
            "return_code": 0,
            "flag_captured": False,
            "done": False,
            "reward": 0.0,
        }

    def step(self, command: str):
        """Execute a command. Must be implemented with an executor."""
        self.step_count += 1
        logger.warning(
            "OpenCTFEnvironment.step() is a stub. "
            "Re-implement with BoxPwnr DockerExecutor for full functionality."
        )
        return {
            "stdout": f"Stub: would execute '{command}'",
            "return_code": 0,
            "flag_captured": False,
            "done": self.step_count >= self.max_steps,
            "reward": 0.0,
            "metadata": {"step": self.step_count},
        }

    def close(self):
        """Clean up resources."""
        pass

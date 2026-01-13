import gymnasium as gym
from gymnasium import spaces
import logging
from typing import Optional, Tuple, Dict, Any
import re
from .docker_manager import DockerManager

logger = logging.getLogger(__name__)

class OpenCTFEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, challenge_id: str, benchmark_path: str = "benchmarks/xbow"):
        """
        Open CTF Environment (OpenEnv Wrapper for XBow).
        
        Args:
            challenge_id: The folder name of the XBow challenge (e.g., 'sqli-login-1')
            benchmark_path: Path to the XBow benchmarks repo
        """
        self.challenge_id = challenge_id
        self.docker_manager = DockerManager(benchmark_path)
        
        # Action Space: Text command (Shell)
        # In Gym, text actions are usually handled as Discrete (if vocab limited) or purely string inputs
        # For OpenEnv/LLM agents, the "action" passed to step() is typically a string or a dict.
        # We declare it as Text for compatibility if using newer Gym versions, otherwise generic.
        self.action_space = spaces.Text(max_length=1000)
        
        # Observation Space: Stdout (Text)
        self.observation_space = spaces.Dict({
            "stdout": spaces.Text(max_length=10000),
            "return_code": spaces.Discrete(256),
            "flag_captured": spaces.Discrete(2) # 0 or 1
        })
        
        self.container_id = None
        self.current_step = 0
        self.max_steps = 50 # Prevent infinite loops
        self.flag_pattern = re.compile(r"(FLAG|XBOW)\{.*?\}") # Adjust based on XBow flag format

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        
        # 1. Cleanup previous run
        if self.container_id:
            self.docker_manager.cleanup()
            
        # 2. Start Challenge & Attacker
        try:
            network_name = self.docker_manager.start_challenge(self.challenge_id)
            self.container_id = self.docker_manager.start_attacker_container(network_name)
            
            # Initial observation
            obs = {
                "stdout": "Environment initialized. Attacker container ready. Type commands to explore.",
                "return_code": 0,
                "flag_captured": 0
            }
            return obs, {}
            
        except Exception as e:
            logger.error(f"Failed to reset environment: {e}")
            raise e

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Executes a shell command in the attacker container.
        """
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        terminated = False
        reward = 0.0
        
        # Execute Command
        stdout, return_code = self.docker_manager.exec_command(self.container_id, action)
        
        # Check for Flag
        flag_match = self.flag_pattern.search(stdout)
        if flag_match:
            reward = 1.0
            terminated = True
            logger.info(f"Flag captured! {flag_match.group(0)}")
        
        # Construct Observation
        obs = {
            "stdout": stdout,
            "return_code": return_code,
            "flag_captured": 1 if reward > 0 else 0
        }
        
        return obs, reward, terminated, truncated, {}

    def render(self):
        # ANSI render likely just handled by the agent's logger
        pass

    def close(self):
        self.docker_manager.cleanup()

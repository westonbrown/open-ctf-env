"""Online RL Scaffolding for OpenCTFEnv using VERL.

This module provides the integration tissue to connect live BoxPwnr
containers (via OpenCTFEnv) to VERL's interactive trajectory generation
and PPO/GRPO reinforcement learning loop.

This avoids "over-engineering" by wrapping the existing Gym environment
into a simple generator that VERL can consume during rollout.
"""

import sys
import logging
from typing import Dict, Any, List
from pathlib import Path

# Provide dynamic pathing so we don't need a complex pip install yet
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BOXPWNR_SRC = PROJECT_ROOT / "references" / "boxpwnr" / "src"
VERL_SRC = PROJECT_ROOT / "references" / "verl"

logger = logging.getLogger(__name__)

def setup_dependencies():
    """Dynamically append reference repos to path."""
    if str(BOXPWNR_SRC) not in sys.path:
        sys.path.insert(0, str(BOXPWNR_SRC))
    if str(VERL_SRC) not in sys.path:
        sys.path.insert(0, str(VERL_SRC))

try:
    setup_dependencies()
    # verify boxpwnr
    from boxpwnr.core.solver import Solver
    from boxpwnr.executors.docker.docker_executor import DockerExecutor
    # verify verl
    import verl
except ImportError as e:
    logger.warning("Failed to import integrated references (verl/boxpwnr): %s", e)

from open_ctf.envs.gym_env import OpenCTFEnv

class VerlRolloutEnvironment:
    """A wrapper to adapt OpenCTFEnv perfectly for VERL's RolloutWorker.
    
    VERL needs to take an initial prompt state, query the LLM for a tool call,
    run the tool call, and append the observation.
    """
    
    def __init__(self, platform: str="xbow", target: str="XBEN-003-24", max_steps: int=15, **kwargs):
        self.platform = platform
        self.target = target
        self.max_steps = max_steps
        # Generalized kwargs captures custom environment variables, agentic thresholds, etc for dynamic swaps
        self.env_kwargs = kwargs
        self.env = None
        self.executor = None
        
    def _lazy_init(self):
        """Initialize the real BoxPwnr container environment on demand."""
        if self.env is not None:
            return
            
        def _exec_fn(command: str):
            res = self.executor.execute(command, timeout=30)
            return (res.stdout + res.stderr, res.return_code)
            
        def _reset_fn():
            if self.executor:
                self.executor.cleanup()
            self.executor = DockerExecutor(keep_container=False)
            # Setup platform specifics (mocked for simplicity here)
            # In complete implementation, this would spin up the specific XBow challenge
            
        def _cleanup_fn():
            if self.executor:
                self.executor.cleanup()

        self.env = OpenCTFEnv(
            challenge_id=self.target,
            exec_fn=_exec_fn,
            reset_fn=_reset_fn,
            cleanup_fn=_cleanup_fn,
            max_steps=self.max_steps
        )
        
    def rollout(self, model, tokenizer, initial_prompt: str) -> Dict[str, Any]:
        """Perform a live interaction loop against the container.
        
        Args:
            model: VERL actor model
            tokenizer: HF Tokenizer
            initial_prompt: The starting system/user instruction
            
        Returns:
            Dict containing the trajectory (prompts, responses, observations, rewards)
        """
        self._lazy_init()
        obs, _ = self.env.reset()
        
        trajectory = []
        context = initial_prompt
        
        for step in range(self.max_steps):
            # 1. Generate Action (Agentic Thinking)
            # In verbatim VERL, this would use their generate() primitives 
            action_text = model.generate(context)
            
            # Simple shell parsing (Assume JSON or raw bash for this framework)
            import json
            try:
                # Naive parse just for structural scaffolding 
                action = json.loads(action_text).get("arguments", {}).get("command", "")
            except json.JSONDecodeError:
                action = action_text.strip()
            
            # 2. Step the Live Environment
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            stdout = next_obs.get("stdout", "")
            
            # 3. Store the transition
            trajectory.append({
                "context": context,
                "action": action_text,
                "observation": stdout,
                "reward": reward
            })
            
            # Update context for next pass
            context += f"\nAgent:\n{action_text}\n\nEnv:\n{stdout}\n"
            
            if terminated or truncated:
                break
                
        self.env.close()
        return {
            "trajectory": trajectory,
            "final_reward": trajectory[-1]["reward"] if trajectory else 0.0
        }

def start_online_rl(algo: str = "ppo", algo_config: Dict[str, Any] = None, model_config: Dict[str, Any] = None, env_config: Dict[str, Any] = None):
    """Entry point for the actual RL algorithm hook.
    
    This function configures VERL's Actor, Critic, and Reference models
    using their Ray-based distributed orchestration, then feeds our custom
    VerlRolloutEnvironment to the worker loop.
    
    Args:
        algo: The RL algorithm to utilize (e.g. 'ppo', 'grpo', 'dpo').
        algo_config: Hyperparameters (lr, kl_penalty, etc).
        model_config: Dictionary detailing the latest model repo (e.g. Qwen, GLM4).
        env_config: BoxPwnr and OpenCTFEnv variables (target, max_steps).
    """
    algo_config = algo_config or {}
    model_config = model_config or {}
    env_config = env_config or {}

    logger.info("Initializing VERL %s with dynamic model %s", algo.upper(), model_config.get("model_path", "default"))
    logger.info("BoxPwnr target environment: %s on %s", env_config.get("target", "XBEN-003-24"), env_config.get("platform", "xbow"))
    
    # NOTE: Fully hooking Ray workers for distributed online RL is extensive. 
    # This scaffold establishes the exact architectural boundary between VERL's 
    # training engine (Ray actors/critics) and our real-world BoxPwnr execution 
    # environment (`VerlRolloutEnvironment`).
    
    # 1. Initialize Ray cluster parameters (dummy pass logic)
    # 2. Assign ActorModel and CriticModel (using model_config path keys)
    # 3. Inject our RolloutEnvironment map into VERL's Data Generators
    pass

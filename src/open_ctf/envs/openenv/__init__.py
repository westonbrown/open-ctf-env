"""OpenEnv-style agent environment with configurable tools.

Server side:
    from open_ctf.envs.openenv import AgentEnvironment, create_app
    app = create_app(target="http://challenge:8080", ground_truth="FLAG{...}")

Client side:
    from open_ctf.envs.openenv import AgentEnv, ToolAction
    env = AgentEnv("http://localhost:8100")
    result = env.reset()
    result = env.step(ToolAction(tool_name="shell_command",
                                  arguments={"command": "id"}))
"""

from .base import Action, Environment, Observation, State, StepResult
from .client import AgentEnv
from .models import ToolAction, ToolObservation, ToolState
from .server import AgentEnvironment, create_app

__all__ = [
    # Base types
    "Action",
    "Observation",
    "State",
    "StepResult",
    "Environment",
    # Concrete types
    "ToolAction",
    "ToolObservation",
    "ToolState",
    # Server
    "AgentEnvironment",
    "create_app",
    # Client
    "AgentEnv",
]

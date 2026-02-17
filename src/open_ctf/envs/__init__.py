"""Open CTF Gymnasium and OpenEnv environments."""

# OpenEnv types (always available)
from .openenv import AgentEnv, AgentEnvironment, ToolAction, ToolObservation, ToolState

__all__ = [
    "AgentEnv",
    "AgentEnvironment",
    "ToolAction",
    "ToolObservation",
    "ToolState",
]

# Gymnasium environment (optional, only if gymnasium is installed)
try:
    from .gym_env import OpenCTFEnv
    __all__.append("OpenCTFEnv")
except ImportError:
    pass

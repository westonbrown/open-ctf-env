from pydantic import Field
from typing import Optional

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    from openenv.core.env_server.types import Action, Observation

class OpenCTFAction(Action):
    """Action for the OpenCTF environment - a shell command."""
    command: str = Field(..., min_length=1, description="Shell command to execute in the attacker container")

class OpenCTFObservation(Observation):
    """Observation from the OpenCTF environment - stdout and status."""
    stdout: str = Field(..., description="Standard output from the command")
    return_code: int = Field(default=0, description="Exit code of the command")
    flag_captured: bool = Field(default=False, description="Whether a flag was captured in this step")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata like step count")

"""Pydantic models for the Open CTF environment."""

from pydantic import BaseModel, Field
from typing import Optional


class OpenCTFAction(BaseModel):
    """Action for the OpenCTF environment - a shell command."""
    command: str = Field(..., min_length=1, description="Shell command to execute in the attacker container")


class OpenCTFObservation(BaseModel):
    """Observation from the OpenCTF environment - stdout and status."""
    stdout: str = Field(..., description="Standard output from the command")
    return_code: int = Field(default=0, description="Exit code of the command")
    flag_captured: bool = Field(default=False, description="Whether a flag was captured in this step")
    done: bool = Field(default=False, description="Whether the episode is done")
    reward: float = Field(default=0.0, description="Reward for this step")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata like step count")

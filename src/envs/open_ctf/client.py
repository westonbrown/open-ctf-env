from typing import Dict, Any

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_server.types import State
    from openenv.core.env_client import EnvClient
    from .models import OpenCTFAction, OpenCTFObservation
except ImportError:
    from openenv.core.client_types import StepResult
    from openenv.core.env_server.types import State
    from openenv.core.env_client import EnvClient
    from models import OpenCTFAction, OpenCTFObservation

class OpenCTFClient(EnvClient[OpenCTFAction, OpenCTFObservation, State]):
    """
    Client for the OpenCTF Environment.
    Connects to the OpenEnv server (locally or remote) to control the CTF agent.
    """

    def _step_payload(self, action: OpenCTFAction) -> Dict:
        return {"command": action.command}

    def _parse_result(self, payload: Dict) -> StepResult[OpenCTFObservation]:
        obs_data = payload.get("observation", {})
        observation = OpenCTFObservation(
            stdout=obs_data.get("stdout", ""),
            return_code=obs_data.get("return_code", 0),
            flag_captured=obs_data.get("flag_captured", False),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

"""
Generator orchestrator for offline synthetic trajectory mapping.
"""

import os
import json
import logging
from typing import Dict, List, Any, Callable
from copy import deepcopy

from litellm import completion

from .manifest import WorldManifest
from .executor import SimulatedEnvironmentExecutor
from ..formatters.tool_registry import BOXPWNR_TOOLS

logger = logging.getLogger(__name__)


class BaseAgentAdapter:
    """
    Interface for hooking up any custom agent scaffolding (LangChain, BoxPwnr, Custom ReAct)
    to the synthetic generation pipeline.
    """
    def run_episode(self, executor: SimulatedEnvironmentExecutor, manifest: WorldManifest, max_turns: int) -> Dict[str, Any]:
        """
        Executes an agent trajectory using the provided executor.
        Must return a trace dictionary matching the SFT trace format:
        { "messages": [...], "metadata": {...}, "ground_truth_flag": "...", "optimal_steps": int }
        """
        raise NotImplementedError


class LiteLLMAgentAdapter(BaseAgentAdapter):
    """
    A simple, zero-shot ReAct-style agent using LiteLLM.
    Acts as the default 'Teacher' model for synthetic generation.
    """
    def __init__(self, model_name: str = "openrouter/openai/gpt-4o"):
        self.model_name = model_name

    def run_episode(self, executor: SimulatedEnvironmentExecutor, manifest: WorldManifest, max_turns: int) -> Dict[str, Any]:
        system_prompt = (
            f"You are a penetration testing agent mapped to the network {manifest.name}. "
            "You have access to execution, file, and meta tools. "
            "Enumerate the environment, identify misconfigurations, and retrieve the flag. "
            "When you find the flag, use the flag_found tool."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Target environment initialized. Objective: Find the flag."}
        ]
        
        step_count = 0
        success = False
        
        while step_count < max_turns:
            step_count += 1
            logger.debug(f"{manifest.name} - Turn {step_count}")
            
            try:
                response = completion(
                    model=self.model_name,
                    messages=messages,
                    tools=BOXPWNR_TOOLS,
                    temperature=0.7,
                )
            except Exception as e:
                logger.error(f"LiteLLM Error: {e}")
                break
                
            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))
            
            if msg.tool_calls:
                for tcall in msg.tool_calls:
                    tool_name = tcall.function.name
                    try:
                        args = json.loads(tcall.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                        
                    res = executor.step(tool_name, args)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tcall.id,
                        "name": tool_name,
                        "content": res["stdout"] or res["stderr"]
                    })
                    
                    if tool_name in ["submit_flag", "flag_found"]:
                        success = (res["exit_code"] == 0)
                        break
            else:
                break
                
            if success or executor._done:
                break
                
        return {
            "messages": messages,
            "metadata": {
                "source": "synthetic_generator",
                "platform": "mock",
                "challenge": manifest.name,
                "success": success,
                "total_turns": step_count,
                "model": self.model_name
            },
            "ground_truth_flag": manifest.ground_truth_flag,
            "optimal_steps": step_count
        }


class BoxPwnrAgentAdapter(BaseAgentAdapter):
    """
    Adapter that hooks the full BoxPwnr ChatCompletionTools strategy 
    into our offline synthetic executor.
    """
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    def run_episode(self, executor: SimulatedEnvironmentExecutor, manifest: WorldManifest, max_turns: int) -> Dict[str, Any]:
        # Here, users load their exact BoxPwnr `Solver` instance.
        # solver = Solver(llm_strategy=ChatCompletionTools(...), executor=executor)
        # solver.solve()
        # trace_data = load_saved_trace()
        # return trace_data
        raise NotImplementedError(
            "BoxPwnrAgentAdapter.run_episode is a scaffold. Provide a concrete "
            "BoxPwnr solver integration before using this adapter."
        )


class SyntheticGenerator:
    """
    Coordinates Generation of synthetic trajectories across mocked environments.
    """
    def __init__(self, manifests: List[WorldManifest], agent_adapter: BaseAgentAdapter):
        self.manifest_bank = manifests
        self.agent_adapter = agent_adapter
        
    def batch_generate_traces(self, max_trajectories: int = 10, max_turns: int = 30) -> List[Dict[str, Any]]:
        generated_traces = []
        for i in range(max_trajectories):
            manifest = self.manifest_bank[i % len(self.manifest_bank)]
            executor = SimulatedEnvironmentExecutor(manifest=manifest, max_steps=max_turns)
            
            # Delegate entirely to the user-supplied agent
            trace = self.agent_adapter.run_episode(executor, manifest, max_turns)
            generated_traces.append(trace)
            
            success = trace.get('metadata', {}).get('success', False)
            logger.info(f"Generated synthetic trace {i+1}/{max_trajectories} -> {manifest.name} (Success: {success})")
            
        return generated_traces

    def enhance_with_reasoning(self, traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        enhanced = []
        for t in traces:
            if not t.get("metadata", {}).get("success", False):
                continue
            trace_copy = deepcopy(t)
            # Retrospective logic goes here
            enhanced.append(trace_copy)
        return enhanced
        
    def export_sft(self, traces: List[Dict[str, Any]], filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            for t in traces:
                f.write(json.dumps(t) + "\n")

    def export_grpo(self, traces: List[Dict[str, Any]], filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            for t in traces:
                f.write(json.dumps(t) + "\n")

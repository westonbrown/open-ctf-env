#!/usr/bin/env python3
"""OpenCTF RL Proxy Runtime Bridge.

This runs a proxy server matching the ROCK architecture. The BYO agent process
is executed in the background (via nohup) and points its API client at the proxy.

The RL generator controls this bridge:
1. It creates the bridge with the external agent command.
2. The agent boots and issues an LLM API request.
3. The bridge intercepts the request and yields it back to the RL generator.
4. The RL generator computes the action + logprobs, sending the action text back.
5. The proxy returns the action text to the agent.
6. This loop repeats until the episode limit or agent exit.

Usage:
  This script is rarely called directly on the CLI. It's imported as a
  module by the SkyRL env or generator to orchestrate episode rollouts.
"""

import asyncio
import json
import logging
import os
import shlex
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

from open_ctf.envs.skyrl.rl_proxy import RLProxyServer

logger = logging.getLogger(__name__)


class RLProxyRuntimeBridge:
    def __init__(self, agent_cmd: str, max_steps: int = 10, port: int = 8000, timeout: int = 600):
        self.agent_cmd = agent_cmd
        self.max_steps = max_steps
        self.timeout = timeout
        
        # Start proxy
        self.proxy = RLProxyServer(port=port)
        
        # Tracking agent state
        self.agent_process: Optional[subprocess.Popen] = None
        self.terminal_reward: float = 0.0
        self.is_done = False
        
        # Setup env passing to child
        self.env = os.environ.copy()
        self.env["OPENAI_API_BASE"] = f"http://localhost:{port}/v1"
        self.env["OPENAI_API_KEY"] = "rl-proxy-key"
        self.env["OPEN_CTF_AGENT_MODE"] = "proxy"

    async def start(self, target: str = "http://localhost:80", challenge_id: str = "Unknown"):
        """Start the proxy server and the background agent process."""
        await self.proxy.start()
        
        # Give the proxy a brief moment to bind to the port before launching agent
        await asyncio.sleep(0.5) 
        
        # Set dynamic environment variables for this episode run
        self.env["OPEN_CTF_TARGET_URL"] = target
        self.env["OPEN_CTF_CHALLENGE_ID"] = challenge_id
        
        logger.info(f"Starting BYO Agent asynchronously: {self.agent_cmd} with target {target}")
        # Execute agent in background. In a full system, we'd use nohup + PID watch,
        # but for this script a subprocess.Popen suffices for prototype.
        self.agent_process = subprocess.Popen(
            shlex.split(self.agent_cmd),
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"Agent started with PID {self.agent_process.pid}")

    async def stop(self):
        """Stop proxy and terminate agent."""
        if self.agent_process:
            if self.agent_process.poll() is None:
                logger.warning(f"Killing agent PID {self.agent_process.pid}")
                self.agent_process.kill()
            self.agent_process = None
            
        await self.proxy.stop()

    async def get_next_prompt(self, wait_timeout: float = 5.0) -> Optional[List[Dict[str, str]]]:
        """Block until the agent sends a prompt to the proxy (or agent exits)."""
        start_time = time.time()
        
        while time.time() - start_time < wait_timeout:
            # Check proxy queue FIRST before checking process exit, 
            # in case the process exited *after* sending the request.
            prompt_req = await self.proxy.yield_prompt(timeout=0.2)
            if prompt_req:
                return prompt_req.get("messages", [])
                
            # Check if agent died/exited
            if self.agent_process and self.agent_process.poll() is not None:
                logger.info("Agent process exited natively.")
                self.is_done = True
                # Parse exit code or stdout for final reward if needed. For now, 0 if 1, etc.
                if self.agent_process.returncode == 0:
                    # Successful exit usually means solved in this setup
                    self.terminal_reward = 1.0
                else:
                    logger.error(f"Agent failed with stderr: {self.agent_process.stderr.read()}")
                return None
                
        # If we hit the outer timeout, check if agent is still alive
        if self.agent_process and self.agent_process.poll() is None:
             logger.error("Agent timeout - hanging without making API call.")
             self.is_done = True
             self.terminal_reward = -1.0 # Timeout penalty
             self.agent_process.kill()
             
        return None

    async def send_action(self, action_text: str):
        """Send the generated action back down to the sleeping proxy response loop."""
        await self.proxy.send_action(action_text)

# Example entrypoint for local testing
if __name__ == "__main__":
    async def main():
        # Example dummy agent cmd
        bridge = RLProxyRuntimeBridge(agent_cmd="python3 -c \"import requests; print(requests.post('http://localhost:8000/v1/chat/completions', json={'messages': [{'role': 'user', 'content': 'hello'}]}).json())\"")
        await bridge.start()
        
        prompt = await bridge.get_next_prompt(wait_timeout=2.0)
        if prompt:
            print("Intercepted:", prompt)
            await bridge.send_action("Hello from RL Proxy!")
            
        await asyncio.sleep(1) # Let callback finish
        await bridge.stop()
        
    asyncio.run(main())

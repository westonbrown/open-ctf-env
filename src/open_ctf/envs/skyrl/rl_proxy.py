import asyncio
import json
import logging
from typing import Any, Dict, Optional, Tuple

from aiohttp import web

logger = logging.getLogger(__name__)

class RLProxyServer:
    """RL-Aware ModelService Proxy.

    Acts as an OpenAI-compatible /v1/chat/completions endpoint for BYO Agents
    running in the sandbox. Instead of processing requests locally, it
    synchronizes with the RL Trainer (e.g. GRPO generator loop), yielding
    the prompt to the GPU and blocking until the action and logprobs are returned.
    """
    def __init__(self, port: int = 8000):
        self.port = port
        self.app = web.Application()
        self.app.router.add_post("/v1/chat/completions", self.handle_chat_completion)
        self.runner: Optional[web.AppRunner] = None
        
        # Synchronization Primitives
        # Queue from Agent -> RL Generator (Prompt requests)
        self._request_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        # Queue from RL Generator -> Agent (Action responses)
        self._response_queue: asyncio.Queue[str] = asyncio.Queue()

    async def start(self):
        """Start the proxy server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '127.0.0.1', self.port)
        await site.start()
        logger.info(f"RL Proxy Server listening on http://127.0.0.1:{self.port}")

    async def stop(self):
        """Stop the proxy server."""
        if self.runner:
            await self.runner.cleanup()

    async def handle_chat_completion(self, request: web.Request) -> web.Response:
        """Handle incoming OpenAI-compatible POST requests from the BYO Agent."""
        try:
            data = await request.json()
            messages = data.get("messages", [])
            
            logger.debug(f"[RLProxy] Received completion request with {len(messages)} messages.")
            
            # Send the request to the RL generator loop
            await self._request_queue.put({
                "messages": messages,
                "model": data.get("model", "default")
            })
            
            # Block until the RL generator computes the action (forward pass)
            action_text = await self._response_queue.get()
            
            # Format as a standard OpenAI response
            response_payload = {
                "id": "chatcmpl-rlproxy",
                "object": "chat.completion",
                "created": 1234567890,
                "model": data.get("model", "default"),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": action_text
                    },
                    "finish_reason": "stop"
                }]
            }
            
            return web.json_response(response_payload)
            
        except Exception as e:
            logger.error(f"[RLProxy] Error handling request: {e}")
            return web.Response(status=500, text=str(e))

    async def yield_prompt(self, timeout: float = 0.5) -> Optional[Dict[str, Any]]:
        """Used by the RL Generator to poll for new prompts from the agent.
        
        Returns None if no prompt is available within the timeout.
        """
        try:
            return await asyncio.wait_for(self._request_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def send_action(self, action_text: str):
        """Used by the RL Generator to send the computed action back to the agent."""
        await self._response_queue.put(action_text)

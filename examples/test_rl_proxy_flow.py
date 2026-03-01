import asyncio
from open_ctf.envs.skyrl.rl_proxy_runtime_bridge import RLProxyRuntimeBridge

async def main():
    bridge = RLProxyRuntimeBridge(
        agent_cmd="python3 /Users/aaronbrown/Downloads/DEV/ghost-training/unprompted/open-ctf-env/examples/mock_langgraph_agent.py",
        port=8001
    )
    
    print("Starting bridge...")
    await bridge.start()
    
    print("Waiting for prompt 1 from agent...")
    prompt1 = await bridge.get_next_prompt(wait_timeout=5.0)
    print("Intercepted Prompt 1:", prompt1)
    
    print("Sending VLLM Action 1 to Agent...")
    await bridge.send_action("<COMMAND>curl -s http://localhost:43008</COMMAND>")
    
    print("Waiting for prompt 2 from agent...")
    prompt2 = await bridge.get_next_prompt(wait_timeout=5.0)
    print("Intercepted Prompt 2:", prompt2)
    
    print("Sending VLLM Action 2 to Agent...")
    await bridge.send_action("<FLAG>captured</FLAG>")
    
    print("Waiting for agent to exit...")
    prompt3 = await bridge.get_next_prompt(wait_timeout=2.0)
    print("Done?", bridge.is_done, "Terminal Reward:", bridge.terminal_reward)
    
    await bridge.stop()

if __name__ == "__main__":
    asyncio.run(main())

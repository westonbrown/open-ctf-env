import json
import random
import argparse
from typing import List, Dict, Any

def inject_failure_loop(tool_call: str, tool_name: str) -> List[Dict[str, Any]]:
    """
    Generates a sequence of (Action -> Failure -> Thought -> Retry) steps.
    """
    loops = []
    
    # Failure Scenarios based on tool type
    if "nmap" in tool_call or "scan" in tool_call:
        failures = [
            ("Connection timed out. Filtered port.", "Firewall blocking standard scans. I need to slow down."),
            ("Host seems down.", "Host might be ignoring ping. I'll try -Pn."),
        ]
        retries = [
            f"{tool_call} -T2", 
            f"{tool_call} -Pn"
        ]
    elif "sqlmap" in tool_call:
        failures = [
            ("403 Forbidden. WAF detected.", "User-Agent blocked. I should randomize it."),
            ("Connection reset by peer.", "Rate limiting detected. Increasing delay."),
        ]
        retries = [
            f"{tool_call} --random-agent",
            f"{tool_call} --delay 2"
        ]
    elif "curl" in tool_call or "http" in tool_call:
        failures = [
            ("404 Not Found", "Endpoint doesn't exist. Let me check robots.txt first."),
            ("500 Internal Server Error", "Payload caused a crash. I need to be more subtle.")
        ]
        retries = [
            f"curl {tool_call.split()[-1]}/robots.txt", # Simplified
            f"{tool_call} --header 'X-Origin: localhost'"
        ]
    else:
        # Generic shell failure
        failures = [
            ("Permission denied.", "I need higher privileges or a different path."),
            ("Command not found.", "Tool missing. I'll check what's installed.")
        ]
        retries = [
            f"sudo {tool_call}",
            "which " + tool_call.split()[0]
        ]

    # Generate 1-3 failure rounds
    num_failures = random.randint(1, 3)
    for i in range(num_failures):
        fail_obs, thought = random.choice(failures)
        retry_cmd = retries[i % len(retries)]
        
        # 1. The original (doomed) action is already "happening" in the main loop context
        # So we insert: Observation(Failure) -> Thought(Recovery) -> Action(Retry)
        
        loops.append({
            "role": "tool",
            "content": f"[ERROR] {fail_obs}",
            "tool_call_id": f"call_{random.randint(1000,9999)}"
        })
        
        loops.append({
            "role": "assistant",
            "content": f"{thought}\n<tool_code>\n{retry_cmd}\n</tool_code>"
        })
        
        # The next loop iteration will provide the result for *this* retry
        # If it's the last failure, the *next* step in the original trace will be the success
        
    return loops

def augment_trace(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Takes a clean success trace and injects failure loops.
    """
    augmented_trace = []
    
    for i, step in enumerate(trace):
        augmented_trace.append(step)
        
        # Only inject failures on Tool Calls (assistant messages with code)
        if step["role"] == "assistant" and "<tool_code>" in step.get("content", ""):
            # 50% chance to inject a failure loop here
            if random.random() < 0.5:
                # Extract the tool command (simplified parsing)
                try:
                    content = step["content"]
                    cmd = content.split("<tool_code>")[1].split("</tool_code>")[0].strip()
                    
                    # Generate failure loop
                    noise = inject_failure_loop(cmd, "shell")
                    augmented_trace.extend(noise)
                    
                    # Note: Ideally we would modify the *next* real observation to match the *last* retry
                    # But for SFT, we just want the model to see the *process* of recovery.
                    # The final "Real Success" from the original trace effectively becomes the result 
                    # of the final "Retry" in our noise loop.
                except:
                    pass
                    
    return augmented_trace

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test_data.jsonl", help="Input clean traces")
    parser.add_argument("--output", default="train_augmented.jsonl", help="Output gritty traces")
    args = parser.parse_args()
    
    # Mock Data Generation for Demo if input missing
    try:
        with open(args.input, 'r') as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("Input file not found. Generating MOCK source data...")
        data = [{
            "messages": [
                {"role": "user", "content": "Scan the target."},
                {"role": "assistant", "content": "Scanning ports.\n<tool_code>\nnmap -p- target\n</tool_code>"},
                {"role": "tool", "content": "Open ports: 80, 22"},
                {"role": "assistant", "content": "Found web server. Enumerating.\n<tool_code>\ngobuster dir -u http://target\n</tool_code>"},
                {"role": "tool", "content": "/admin found"},
                {"role": "assistant", "content": "Checking admin.\n<tool_code>\ncurl http://target/admin\n</tool_code>"},
                {"role": "tool", "content": "XBOW{FLAG}"}
            ]
        }]

    with open(args.output, 'w') as f:
        for trace in data:
            # Augment the message list
            orig_msgs = trace["messages"]
            aug_msgs = augment_trace(orig_msgs)
            
            # Write out
            json.dump({"messages": aug_msgs}, f)
            f.write('\n')
            
    print(f"Generated {len(data)} augmented traces to {args.output}")

if __name__ == "__main__":
    main()

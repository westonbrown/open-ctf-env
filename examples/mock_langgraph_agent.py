import urllib.request
import urllib.error
import urllib.parse
import json
import os
import sys
import time

def post_json(url, data, timeout=5):
    req = urllib.request.Request(
        url, 
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))

def main():
    base_url = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
    
    print(f"[MockAgent] Starting up. API Base: {base_url}")
    time.sleep(1) # mock init time
    
    payload = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a BoxPwnr autonomous agent."},
            {"role": "user", "content": "Begin exploration of http://localhost:43008"}
        ]
    }
    
    print("[MockAgent] Sending prompt 1...")
    try:
        resp = post_json(f"{base_url}/chat/completions", payload, timeout=5)
        print("[MockAgent] Received response:", resp)
        
        # Parse action mock -> pretend to execute tool -> get result
        action_text = resp["choices"][0]["message"]["content"]
        
        # Send follow up
        payload["messages"].append({"role": "assistant", "content": action_text})
        payload["messages"].append({"role": "user", "content": "Tool executed successfully. <FLAG>you_found_me</FLAG>"})
        
        print("[MockAgent] Sending prompt 2...")
        resp2 = post_json(f"{base_url}/chat/completions", payload, timeout=5)
        print("[MockAgent] Received response 2:", resp2)
        
    except Exception as e:
        print(f"[MockAgent] Request failed: {e}")
        sys.exit(1)
        
    print("[MockAgent] Done. Exiting 0.")
    sys.exit(0)

if __name__ == "__main__":
    main()

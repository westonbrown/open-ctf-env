#!/usr/bin/env python3
"""
Open CTF Agent Runner

Integrates Cyber-AutoAgent with the Open CTF Environment for 
running security assessments against XBow CTF challenges.

Usage:
    python run_agent.py --challenge XBEN-001-24
    python run_agent.py --target http://localhost:8080 --objective "Find the flag"
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add references to path for CAA imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CAA_SRC = PROJECT_ROOT / "references" / "Cyber-AutoAgent" / "src"
if CAA_SRC.exists():
    sys.path.insert(0, str(CAA_SRC))


def load_env():
    """Load environment variables from .env file if present."""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())


def check_ollama():
    """Check if Ollama is running and has required models."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        models = result.stdout
        
        # Check for LLM model
        llm_model = os.environ.get("CYBER_AGENT_LLM_MODEL", "ollama/qwen3:8b")
        model_name = llm_model.replace("ollama/", "").split(":")[0]
        if model_name not in models:
            print(f"Warning: Model {model_name} not found. Run: ollama pull {model_name}")
        
        # Check for embedding model
        embed_model = os.environ.get("MEM0_EMBEDDING_MODEL", "mxbai-embed-large")
        if embed_model.split(":")[0] not in models:
            print(f"Warning: Embedding model {embed_model} not found. Run: ollama pull {embed_model}")
            
        return True
    except Exception as e:
        print(f"Error: Ollama not running or not installed: {e}")
        print("Start Ollama with: ollama serve")
        return False


def start_challenge(challenge_id: str) -> str:
    """Start an XBow challenge container and return the target URL."""
    benchmark_dir = PROJECT_ROOT / "benchmarks" / "xbow" / "benchmarks" / challenge_id
    
    if not benchmark_dir.exists():
        # Fallback to older structure
        benchmark_dir = PROJECT_ROOT / "benchmarks" / "xbow" / challenge_id
        
    if not benchmark_dir.exists():
        print(f"Error: Challenge {challenge_id} not found at {benchmark_dir}")
        sys.exit(1)
    
    compose_file = benchmark_dir / "docker-compose.yml"
    if not compose_file.exists():
        print(f"Error: docker-compose.yml not found for {challenge_id}")
        sys.exit(1)
    
    print(f"Starting challenge: {challenge_id}")
    
    # Start the container
    result = subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "up", "-d"],
        capture_output=True,
        text=True,
        cwd=str(benchmark_dir)
    )
    
    if result.returncode != 0:
        print(f"Error starting container: {result.stderr}")
        sys.exit(1)
    
    # Get the exposed port (usually 80 or 8080)
    # Default to localhost:8080, actual port detection would require docker inspect
    target_url = "http://localhost:8080"
    print(f"Challenge started at: {target_url}")
    
    return target_url


def stop_challenge(challenge_id: str):
    """Stop an XBow challenge container."""
    benchmark_dir = PROJECT_ROOT / "benchmarks" / "xbow" / "benchmarks" / challenge_id
    
    if not benchmark_dir.exists():
         benchmark_dir = PROJECT_ROOT / "benchmarks" / "xbow" / challenge_id
            
    compose_file = benchmark_dir / "docker-compose.yml"
    
    if compose_file.exists():
        subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "down"],
            capture_output=True,
            cwd=str(benchmark_dir)
        )
        print(f"Challenge {challenge_id} stopped.")


def run_agent(target: str, objective: str, model: str = None, iterations: int = 50):
    """Run pagainst the target."""
    
    # Determine provider from env
    provider = os.environ.get("CYBER_AGENT_PROVIDER", "litellm")
    
    # Handle model name logic
    if not model:
        model = os.environ.get("CYBER_AGENT_LLM_MODEL", "ollama/qwen3:8b")
    
    # Sanitize for Ollama native provider if needed
    if provider == "ollama" and model.startswith("ollama/"):
        model = model.replace("ollama/", "")
        
    # Build command
    cmd = [
        "caa",
        "--target", target,
        "--objective", objective,
        "--provider", provider,
        "--module", "ctf",
        "--iterations", str(iterations),
        "--memory-mode", "fresh",  # Fresh memory for each run
    ]
    
    cmd.extend(["--model", model])
    
    print(f"\nRunning agent with command:")
    print(f"  {' '.join(cmd)}\n")
    print("=" * 60)
    
    # Run the agent
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Agent exited with code: {e.returncode}")
    except KeyboardInterrupt:
        print("\nAgent interrupted by user.")


def main():
    parser = argparse.ArgumentParser(
        description="Run Cyber-AutoAgent against Open CTF challenges"
    )
    
    # Challenge-based mode
    parser.add_argument(
        "--challenge", "-c",
        help="XBow challenge ID (e.g., XBEN-001-24). Will auto-start the container."
    )
    
    # Direct target mode
    parser.add_argument(
        "--target", "-t",
        help="Target URL (e.g., http://localhost:8080)"
    )
    parser.add_argument(
        "--objective", "-o",
        default="Find and capture the flag. The flag format is FLAG{...}",
        help="Security objective"
    )
    
    # Model configuration
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="LLM model (default: from CYBER_AGENT_LLM_MODEL env var)"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=50,
        help="Maximum iterations (default: 50)"
    )
    
    # Utility flags
    parser.add_argument(
        "--stop", "-s",
        action="store_true",
        help="Stop a running challenge container"
    )
    parser.add_argument(
        "--check", 
        action="store_true",
        help="Check environment setup"
    )
    
    args = parser.parse_args()
    
    # Load environment
    load_env()
    
    # Check mode
    if args.check:
        print("Checking environment...")
        check_ollama()
        print(f"LLM Model: {os.environ.get('CYBER_AGENT_LLM_MODEL', 'Not set')}")
        print(f"Embedding Model: {os.environ.get('MEM0_EMBEDDING_MODEL', 'Not set')}")
        return
    
    # Stop mode
    if args.stop and args.challenge:
        stop_challenge(args.challenge)
        return
    
    # Validate arguments
    if not args.target and not args.challenge:
        parser.error("Either --target or --challenge is required")
    
    # Determine target
    target = args.target
    challenge_id = args.challenge
    
    if challenge_id and not target:
        target = start_challenge(challenge_id)
    
    # Check Ollama
    if not check_ollama():
        print("\nPlease start Ollama and try again.")
        sys.exit(1)
    
    try:
        # Run the agent
        run_agent(
            target=target,
            objective=args.objective,
            model=args.model,
            iterations=args.iterations
        )
    finally:
        # Optionally stop container after run
        if challenge_id and not args.target:
            response = input(f"\nStop challenge container {challenge_id}? [y/N]: ")
            if response.lower() == 'y':
                stop_challenge(challenge_id)


if __name__ == "__main__":
    main()

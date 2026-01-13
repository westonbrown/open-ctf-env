import subprocess
import time
import os
import logging
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

class DockerManager:
    def __init__(self, benchmark_path: str = "benchmarks/xbow/benchmarks"):
        """
        Manages Docker containers for XBow challenges.
        
        Args:
            benchmark_path: Path to the cloned xbow-validation-benchmarks repo (subdir containing challenges)
        """
        self.benchmark_path = os.path.abspath(benchmark_path)
        self.active_containers: Dict[str, str] = {}  # challenge_id -> container_id

    def start_challenge(self, challenge_id: str) -> str:
        """
        Starts the challenge containers using docker compose.
        Returns the network name to attach the attacker to.
        """
        challenge_dir = os.path.join(self.benchmark_path, challenge_id)
        if not os.path.exists(challenge_dir):
            # Fallback check for missing 'benchmarks' subdir if user pointed to root
            if "benchmarks" not in self.benchmark_path:
                 alt_path = os.path.join(self.benchmark_path, "benchmarks", challenge_id)
                 if os.path.exists(alt_path):
                     challenge_dir = alt_path
            
            if not os.path.exists(challenge_dir):
                raise ValueError(f"Challenge {challenge_id} not found at {challenge_dir}")
            
        compose_file = os.path.join(challenge_dir, "docker-compose.yml")
        if not os.path.exists(compose_file):
            raise ValueError(f"docker-compose.yml not found for {challenge_id}")

        logger.info(f"Starting challenge {challenge_id}...")
        
        # 1. Start Challenge via Docker Compose V2
        subprocess.run(
            ["docker", "compose", "-f", compose_file, "up", "-d"],
            check=True,
            capture_output=True
        )
        
        # 2. Identify the network name
        # TODO: Parse docker-compose.yml or inspect running container to find actual network
        # For XBow, many use explicit network names (e.g. xben-001-network)
        # Fallback to guessing if not easily determinable without deps
        network_name = f"{challenge_id.lower()}-network"  # Common pattern in XBow

        
        return network_name

    def start_attacker_container(self, network_name: str) -> str:
        """
        Starts a Kali/Alpine attacker container attached to the challenge network.
        Returns the container ID.
        """
        image_name = "kalilinux/kali-rolling" # Or use a lighter alpine with tools installed
        container_name = f"openctf_attacker_{int(time.time())}"
        
        # Ensure we have the image
        # subprocess.run(["docker", "pull", image_name], check=True)

        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "--network", network_name,
            "--entrypoint", "tail",
            image_name,
            "-f", "/dev/null" # Keep alive
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        container_id = result.stdout.strip()
        self.active_containers[container_name] = container_id
        
        # Pre-install essential tools if using a bare image (or use a pre-built image)
        # self.exec_command(container_id, "apt-get update && apt-get install -y curl nmap sqlmap")
        
        return container_id

    def exec_command(self, container_id: str, command: str, timeout: int = 60) -> Tuple[str, int]:
        """
        Executes a command in the container.
        Returns (stdout, return_code).
        """
        try:
            # Using docker exec
            # strict_mode: bash -c 'command' to handle pipes/redirects
            cmd = ["docker", "exec", container_id, "bash", "-c", command]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.stdout + result.stderr, result.returncode
            
        except subprocess.TimeoutExpired:
            return "Command timed out", 124
        except Exception as e:
            return str(e), 1

    def stop_challenge(self, challenge_id: str):
        """
        Stops the challenge and attacker containers.
        """
        challenge_dir = os.path.join(self.benchmark_path, challenge_id)
        compose_file = os.path.join(challenge_dir, "docker-compose.yml")
        
        # Down compose
        if os.path.exists(compose_file):
            subprocess.run(
                ["docker", "compose", "-f", compose_file, "down", "-v"],
                capture_output=True
            )
            
        # Kill attacker containers? (We need to track which attacker belongs to which challenge)
        # For now, we'll just implement a global cleanup or simple tracking if needed.
        pass

    def cleanup(self):
        """Force cleanup of all tracked containers"""
        for name, cid in self.active_containers.items():
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True)

"""Challenge lifecycle manager — launch/stop Docker containers for CTF challenges."""

import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from .registry import ChallengeRegistry

logger = logging.getLogger(__name__)


class ChallengeManager:
    """Manage Docker container lifecycle for benchmark challenges.

    Follows BoxPwnr's CybenchPlatform pattern:
    1. Look up challenge in registry
    2. Run init_script.sh if present
    3. docker compose up -d
    4. Health check
    5. Return target URL

    Usage::

        registry = ChallengeRegistry("configs/challenges/cybench.yaml")
        manager = ChallengeManager(registry, bench_dir="/path/to/cybench")
        url = manager.setup("eval-me")  # -> "http://localhost:32805"
        manager.teardown("eval-me")
    """

    def __init__(
        self,
        registry: ChallengeRegistry,
        bench_dir: str,
        host: str = "localhost",
        network: str = "shared_net",
    ):
        self.registry = registry
        self.bench_dir = Path(bench_dir)
        self.host = host
        self.network = network
        self._running: Dict[str, str] = {}  # challenge_id -> target_url

    def _challenge_dir(self, challenge_id: str) -> Path:
        """Get the directory for a challenge in the benchmark repo."""
        return self.bench_dir / "benchmark" / challenge_id

    def setup(self, challenge_id: str) -> str:
        """Launch a challenge's Docker container and return the target URL.

        Args:
            challenge_id: Challenge identifier from the registry.

        Returns:
            Target URL (e.g. "http://localhost:32805").

        Raises:
            KeyError: If challenge not in registry.
            ValueError: If challenge is static (no Docker needed).
            RuntimeError: If container fails to start.
        """
        info = self.registry.get(challenge_id)

        if info.infra_type != "docker":
            raise ValueError(
                f"Challenge {challenge_id} is {info.infra_type}, not docker — no container to launch"
            )

        challenge_dir = self._challenge_dir(challenge_id)

        # Run init_script.sh if present (builds images, etc.)
        init_script = challenge_dir / "init_script.sh"
        if init_script.exists():
            logger.info("Running init script for %s", challenge_id)
            result = subprocess.run(
                ["bash", str(init_script)],
                cwd=str(challenge_dir),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                logger.warning("init_script.sh failed for %s: %s", challenge_id, result.stderr)

        # Start with docker compose
        compose_file = challenge_dir / "docker-compose.yaml"
        if not compose_file.exists():
            compose_file = challenge_dir / "docker-compose.yml"

        if compose_file.exists():
            logger.info("Starting docker compose for %s", challenge_id)
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_file), "up", "-d"],
                cwd=str(challenge_dir),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"docker compose up failed for {challenge_id}: {result.stderr}"
                )
        else:
            # Try start_docker.sh fallback
            start_script = challenge_dir / "start_docker.sh"
            if start_script.exists():
                logger.info("Running start_docker.sh for %s", challenge_id)
                result = subprocess.run(
                    ["bash", str(start_script)],
                    cwd=str(challenge_dir),
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"start_docker.sh failed for {challenge_id}: {result.stderr}"
                    )
            else:
                raise RuntimeError(
                    f"No docker-compose.yaml or start_docker.sh found in {challenge_dir}"
                )

        target_url = self.registry.get_target_url(challenge_id, host=self.host)
        if target_url:
            self._running[challenge_id] = target_url

        return target_url or f"http://{self.host}:{info.port}"

    def teardown(self, challenge_id: str) -> None:
        """Stop a challenge's Docker container.

        Args:
            challenge_id: Challenge identifier.
        """
        info = self.registry.get(challenge_id)
        if info.infra_type != "docker":
            return

        challenge_dir = self._challenge_dir(challenge_id)

        compose_file = challenge_dir / "docker-compose.yaml"
        if not compose_file.exists():
            compose_file = challenge_dir / "docker-compose.yml"

        if compose_file.exists():
            logger.info("Stopping docker compose for %s", challenge_id)
            subprocess.run(
                ["docker", "compose", "-f", str(compose_file), "down"],
                cwd=str(challenge_dir),
                capture_output=True,
                text=True,
                timeout=60,
            )

        self._running.pop(challenge_id, None)

    def setup_all(self, ids: Optional[List[str]] = None) -> Dict[str, str]:
        """Launch multiple challenges. Returns {challenge_id: target_url}.

        Args:
            ids: Specific challenge IDs. If None, launches all docker challenges.
        """
        if ids is None:
            ids = [c.id for c in self.registry.list_docker_challenges()]

        results = {}
        for cid in ids:
            try:
                url = self.setup(cid)
                results[cid] = url
            except Exception as exc:
                logger.error("Failed to setup %s: %s", cid, exc)
        return results

    def teardown_all(self) -> None:
        """Stop all running challenge containers."""
        for cid in list(self._running.keys()):
            try:
                self.teardown(cid)
            except Exception as exc:
                logger.error("Failed to teardown %s: %s", cid, exc)

    def health_check(self, challenge_id: str, timeout: int = 5) -> bool:
        """Check if a challenge's service is responding.

        Args:
            challenge_id: Challenge identifier.
            timeout: HTTP timeout in seconds.

        Returns:
            True if service responds, False otherwise.
        """
        url = self.registry.get_target_url(challenge_id, host=self.host)
        if not url:
            return False

        try:
            result = subprocess.run(
                ["curl", "-sf", "--max-time", str(timeout), "-o", "/dev/null", "-w", "%{http_code}", url],
                capture_output=True,
                text=True,
                timeout=timeout + 5,
            )
            status = result.stdout.strip()
            return status.startswith(("2", "3", "4"))  # Any HTTP response = service is up
        except (subprocess.TimeoutExpired, Exception):
            return False

    def get_running(self) -> List[str]:
        """Return list of currently running challenge IDs."""
        return list(self._running.keys())

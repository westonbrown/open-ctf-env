"""BoxPwnr-based CTF agent runner.

Wraps BoxPwnr's Solver to run CTF challenges using LLM agents.
Requires the BoxPwnr reference repo at references/boxpwnr/.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

BOXPWNR_SRC = Path(__file__).resolve().parents[3] / "references" / "boxpwnr" / "src"

logger = logging.getLogger(__name__)


def _import_boxpwnr():
    """Import BoxPwnr components. Raises ImportError with guidance if missing."""
    # Only add BoxPwnr to path when actually needed (avoids import conflicts)
    if BOXPWNR_SRC.exists() and str(BOXPWNR_SRC) not in sys.path:
        sys.path.insert(0, str(BOXPWNR_SRC))
    try:
        from boxpwnr.core.solver import Solver
        from boxpwnr.executors.docker.docker_executor import DockerExecutor
        from boxpwnr.strategies import ChatCompletionToolsStrategy, ChatCompletionStrategy
        from boxpwnr.utils.secrets_manager import SecretManager
        return Solver, DockerExecutor, ChatCompletionToolsStrategy, ChatCompletionStrategy, SecretManager
    except ImportError as e:
        raise ImportError(
            f"BoxPwnr not found at {BOXPWNR_SRC}. "
            f"Ensure references/boxpwnr/ exists with the full BoxPwnr repo. "
            f"Original error: {e}"
        ) from e


def _get_platform(platform_name: str, executor, traces_dir: str, keep_target: bool = False):
    """Create a BoxPwnr platform instance by name."""
    if platform_name == "xbow":
        from boxpwnr.platforms.xbow import XBOWPlatform
        return XBOWPlatform(executor=executor, traces_dir=traces_dir, keep_target=keep_target)
    elif platform_name == "local":
        from boxpwnr.platforms.local import LocalPlatform
        return LocalPlatform(executor=executor, traces_dir=traces_dir, keep_target=keep_target)
    elif platform_name == "htb":
        from boxpwnr.platforms.htb import HTBPlatform
        return HTBPlatform(executor=executor, traces_dir=traces_dir, keep_target=keep_target)
    elif platform_name == "portswigger":
        from boxpwnr.platforms.portswigger import PortSwiggerPlatform
        return PortSwiggerPlatform(executor=executor, traces_dir=traces_dir, keep_target=keep_target)
    elif platform_name == "cybench":
        from boxpwnr.platforms.cybench import CybenchPlatform
        return CybenchPlatform(executor=executor, traces_dir=traces_dir, keep_target=keep_target)
    else:
        raise ValueError(
            f"Unknown platform: {platform_name}. "
            f"Supported: xbow, local, htb, portswigger, cybench"
        )


class AgentRunner:
    """Runs BoxPwnr solver against CTF challenges.

    This is a thin wrapper around BoxPwnr's Solver that provides
    a simplified interface for the Open CTF project.

    Usage:
        runner = AgentRunner(platform="xbow", model="ollama/glm-4.7-flash")
        runner.run(target="XBEN-003-24")
    """

    def __init__(
        self,
        platform: str = "xbow",
        model: str = "openrouter/openai/gpt-oss-120b",
        strategy: str = "chat_tools",
        max_turns: int = 50,
        max_time: Optional[int] = 30,
        max_cost: Optional[float] = None,
        traces_dir: str = "./targets",
        debug: bool = False,
        keep_container: bool = False,
        keep_target: bool = False,
        reasoning_effort: str = "medium",
        attempts: int = 1,
        custom_instructions: Optional[str] = None,
    ):
        """Initialize the agent runner.

        Args:
            platform: Target platform (xbow, local, htb, portswigger, cybench).
            model: LLM model identifier (e.g. openrouter/openai/gpt-oss-120b).
            strategy: LLM strategy (chat, chat_tools).
            max_turns: Maximum conversation turns per attempt.
            max_time: Maximum time in minutes per attempt.
            max_cost: Maximum cost in USD per attempt.
            traces_dir: Directory to store trace artifacts.
            debug: Enable debug logging.
            keep_container: Keep Docker container after completion.
            keep_target: Keep target running after completion.
            reasoning_effort: Reasoning effort level for supported models.
            attempts: Number of solve attempts.
            custom_instructions: Additional instructions appended to system prompt.
        """
        self.platform_name = platform
        self.model = model
        self.strategy_name = strategy
        self.max_turns = max_turns
        self.max_time = max_time
        self.max_cost = max_cost
        self.traces_dir = traces_dir
        self.debug = debug
        self.keep_container = keep_container
        self.keep_target = keep_target
        self.reasoning_effort = reasoning_effort
        self.attempts = attempts
        self.custom_instructions = custom_instructions

    def check_setup(self) -> bool:
        """Verify BoxPwnr components can be imported.

        Returns:
            True if all imports succeed, False otherwise.
        """
        try:
            _import_boxpwnr()
            print(f"BoxPwnr source: {BOXPWNR_SRC}")
            print(f"Platform:       {self.platform_name}")
            print(f"Model:          {self.model}")
            print(f"Strategy:       {self.strategy_name}")
            print("All components OK.")
            return True
        except ImportError as e:
            print(f"Setup check failed: {e}")
            return False

    def run(self, target: str):
        """Run the solver against a target.

        Args:
            target: Target identifier (e.g. XBEN-003-24 for xbow).
        """
        Solver, DockerExecutor, ChatCompletionToolsStrategy, ChatCompletionStrategy, SecretManager = _import_boxpwnr()

        # Build traces dir with platform subdirectory
        traces_dir = f"{self.traces_dir}/{self.platform_name}"

        # Create executor
        executor = DockerExecutor(
            keep_container=self.keep_container,
            default_timeout=30,
            max_timeout=300,
            use_interactive_sessions=(self.strategy_name == "chat_tools"),
        )

        # Create platform
        platform = _get_platform(
            self.platform_name,
            executor=executor,
            traces_dir=traces_dir,
            keep_target=self.keep_target,
        )

        # Create secrets manager
        secrets_manager = SecretManager()

        # Create LLM strategy
        if self.strategy_name == "chat_tools":
            llm_strategy = ChatCompletionToolsStrategy(
                model=self.model,
                secrets_manager=secrets_manager,
                executor=executor,
                reasoning_effort=self.reasoning_effort,
            )
        elif self.strategy_name == "chat":
            llm_strategy = ChatCompletionStrategy(
                model=self.model,
                secrets_manager=secrets_manager,
                reasoning_effort=self.reasoning_effort,
            )
        else:
            raise ValueError(
                f"Unknown strategy: {self.strategy_name}. Supported: chat, chat_tools"
            )

        # Create and run solver
        solver = Solver(
            target_name=target,
            platform=platform,
            executor=executor,
            llm_strategy=llm_strategy,
            traces_dir=traces_dir,
            strategy_name=self.strategy_name,
            debug=self.debug,
            max_turns=self.max_turns,
            max_cost=self.max_cost,
            max_time=self.max_time,
            attempts=self.attempts,
            custom_instructions=self.custom_instructions,
        )

        solver.solve()

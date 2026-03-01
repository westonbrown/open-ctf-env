"""Open CTF environments."""
from .tool_executor import SubprocessExecutor

__all__ = ["SubprocessExecutor"]

# parse_tool_calls lives in open_ctf.parsing but is re-exported here for
# backward compatibility with code that imports from open_ctf.envs.
try:
    from open_ctf.parsing import parse_tool_calls
    __all__.append("parse_tool_calls")
except ImportError:
    pass

# SkyRL env (optional)
try:
    from .skyrl.openctf_env import OpenCTFTextEnv
    __all__.append("OpenCTFTextEnv")
except ImportError:
    pass

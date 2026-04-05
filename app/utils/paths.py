import os
from pathlib import Path


def get_base_dir() -> Path:
    """Returns the base directory of the project."""
    return Path(__file__).resolve().parent.parent.parent


def env_rel_path(env_key: str, default_rel_path: str) -> Path:
    """Resolve an environment-configured path relative to the base directory."""
    val = os.getenv(env_key)
    if val:
        return Path(val).resolve()
    return get_base_dir() / default_rel_path


def get_depth_repo_path() -> str:
    """
    Get the Depth-Anything-V2 repository path, removing the hardcoded Windows path.
    """
    repo_path = os.getenv("DEPTH_ANYTHING_V2_REPO")
    if repo_path:
        return repo_path

    base_dir = get_base_dir()
    default_path = base_dir / "Depth-Anything-V2"
    return str(default_path)

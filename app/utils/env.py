import os
from app.utils.paths import get_base_dir


def load_env_file(env_file_path: str = None) -> None:
    """Load an environment file and inject variables into os.environ."""
    if env_file_path is None:
        env_file_path = str(get_base_dir() / ".env")

    if not os.path.exists(env_file_path):
        return

    with open(env_file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

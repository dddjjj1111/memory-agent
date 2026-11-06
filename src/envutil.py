"""Environment helpers. The optional ``.env`` loader is stdlib-only."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs into the environment.

    Existing variables win, so a shell export or a test monkeypatch is not
    overwritten by a local file. Missing files are ignored.
    """

    env_path = path or Path(".env")
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def use_mock_llm() -> bool:
    """Default is the offline heuristic so a fresh clone runs without a key."""

    raw = os.environ.get("USE_MOCK_LLM", "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default

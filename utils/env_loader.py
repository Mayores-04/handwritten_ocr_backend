"""Minimal .env.local loader for local development."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def _parse_env_lines(lines: Iterable[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        env[key] = value
    return env


def load_env_local(root: Path | None = None, filename: str = ".env.local") -> None:
    root_path = root or Path(__file__).resolve().parent.parent
    env_path = root_path / filename
    if not env_path.exists():
        return

    env_values = _parse_env_lines(env_path.read_text(encoding="utf-8").splitlines())
    for key, value in env_values.items():
        os.environ.setdefault(key, value)

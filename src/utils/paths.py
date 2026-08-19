"""Resolve project-relative paths from dissertation_project/ root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the dissertation_project/ directory."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve a path relative to the project root unless absolute."""
    root = root or project_root()
    p = Path(path)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    cfg_path = resolve_path(path) if not Path(path).is_absolute() else Path(path)
    if not cfg_path.exists():
        # Allow loading relative to project root via configs/
        alt = project_root() / path
        cfg_path = alt if alt.exists() else cfg_path
    with open(cfg_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if needed and return it."""
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

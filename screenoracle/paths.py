from __future__ import annotations

from pathlib import Path


def get_project_root(start: Path | None = None) -> Path:
    """Return repository root assuming code lives in screenoracle/ under root."""
    here = (start or Path(__file__)).resolve()
    return here.parents[1]


def get_imdb_dir(project_root: Path | None = None, *, imdb_subdir: str) -> Path:
    root = project_root or get_project_root()
    return (root / imdb_subdir).resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

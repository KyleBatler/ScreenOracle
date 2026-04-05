from __future__ import annotations

from pathlib import Path

import pandas as pd

from screenoracle.i18n import Lang, t
from screenoracle.paths import get_project_root


def read_imdb_gz_table(
    file_path: Path,
    *,
    dtype: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load a single IMDb `.tsv.gz` table into a DataFrame."""
    return pd.read_csv(
        file_path,
        sep="\t",
        na_values="\\N",
        dtype=dtype,
        low_memory=False,
        compression="gzip",
    )


def get_default_imdb_paths(
    *,
    imdb_subdir: str,
    basics_name: str,
    ratings_name: str,
    project_root: Path | None = None,
) -> tuple[Path, Path]:
    root = project_root or get_project_root()
    imdb_dir = (root / imdb_subdir).resolve()
    basics_path = imdb_dir / basics_name
    ratings_path = imdb_dir / ratings_name
    return basics_path, ratings_path


def assert_files_non_empty(paths: list[Path], *, lang: Lang) -> None:
    missing = [p for p in paths if not p.is_file()]
    if missing:
        joined = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(t(lang, "err_missing_file", paths=joined))
    empty = [p for p in paths if p.stat().st_size == 0]
    if empty:
        joined = "\n".join(str(p) for p in empty)
        raise ValueError(f"These IMDb files are empty (0 bytes):\n{joined}")


def load_title_basics(basics_path: Path) -> pd.DataFrame:
    return read_imdb_gz_table(
        basics_path,
        dtype={
            "tconst": "string",
            "titleType": "string",
            "primaryTitle": "string",
            "originalTitle": "string",
            "isAdult": "string",
        },
    )


def load_title_ratings(ratings_path: Path) -> pd.DataFrame:
    return read_imdb_gz_table(
        ratings_path,
        dtype={"tconst": "string"},
    )

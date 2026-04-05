from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScreenOracleConfig:
    """Central place for defaults. Tune deliberately; document changes in runs."""

    imdb_subdir: str = "data/raw/imdb"
    basics_name: str = "title.basics.tsv.gz"
    ratings_name: str = "title.ratings.tsv.gz"

    train_year_end: int = 2009
    test_year_start: int = 2010
    test_year_end: int = 2019

    min_votes_for_label: int = 500
    rating_threshold: float = 7.0

    top_genres: int = 25
    random_state: int = 42

    model_n_estimators: int = 200
    model_max_depth: int | None = 12
    model_min_samples_leaf: int = 5
    model_n_jobs: int = -1

    models_dir: str = "models"
    artifact_name: str = "screenoracle_forest.joblib"
    threshold_name: str = "decision_threshold.txt"


DEFAULT_CONFIG = ScreenOracleConfig()

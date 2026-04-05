from __future__ import annotations

import numpy as np
import pandas as pd

from screenoracle.config import ScreenOracleConfig


def drop_duplicate_tconst(df: pd.DataFrame, *, table_name: str) -> pd.DataFrame:
    """IMDb keys should be unique; if not, keep the first row and continue with a warning count."""
    dup = int(df["tconst"].duplicated().sum())
    if dup:
        # Not print here: keep this function pure; caller can log.
        pass
    out = df.drop_duplicates(subset=["tconst"], keep="first").copy()
    out.attrs["duplicates_dropped"] = dup
    out.attrs["table_name"] = table_name
    return out


def merge_basics_and_ratings(basics: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Inner-merge titles with ratings on `tconst` (only rated rows remain)."""
    b = drop_duplicate_tconst(basics, table_name="title.basics")
    r = drop_duplicate_tconst(ratings, table_name="title.ratings")
    return b.merge(r, on="tconst", how="inner")


def keep_movies_only(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["titleType"] == "movie"].copy()


def coerce_core_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["startYear"] = pd.to_numeric(out["startYear"], errors="coerce")
    out["runtimeMinutes"] = pd.to_numeric(out["runtimeMinutes"], errors="coerce")
    out["averageRating"] = pd.to_numeric(out["averageRating"], errors="coerce")
    out["numVotes"] = pd.to_numeric(out["numVotes"], errors="coerce")
    out["isAdult"] = pd.to_numeric(out["isAdult"], errors="coerce").fillna(0).astype("int8")
    return out


def extract_primary_genre(genres: pd.Series) -> pd.Series:
    """IMDb `genres` is comma-separated; take first label, else Unknown."""

    def one(value: object) -> str:
        if pd.isna(value):
            return "Unknown"
        parts = str(value).split(",")
        first = parts[0].strip()
        return first if first else "Unknown"

    return genres.map(one).astype("string")


def collapse_rare_genres(genres: pd.Series, allowed: set[str]) -> pd.Series:
    def one(g: object) -> str:
        s = str(g)
        return s if s in allowed else "Other"

    return genres.map(one).astype("string")


def fit_top_genre_allowlist(train_genres: pd.Series, *, top_k: int) -> set[str]:
    counts = train_genres.value_counts()
    head = counts.head(top_k)
    return set(head.index.astype(str))


def add_success_label(
    df: pd.DataFrame,
    *,
    min_votes: int,
    rating_threshold: float,
) -> pd.DataFrame:
    """
    Binary label for historical supervised learning.

    'success' here means: enough votes to be somewhat stable AND rating at/above threshold.

    This is not box office; IMDb dumps do not include worldwide gross.
    """
    out = df.copy()
    eligible = out["numVotes"] >= min_votes
    high_rating = out["averageRating"] >= rating_threshold
    out["success"] = (eligible & high_rating).astype("int8")
    return out


def drop_unusable_rows_for_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot be featurized (missing year)."""
    out = df.copy()
    out = out.loc[out["startYear"].notna()].copy()
    return out


def temporal_masks(
    years: pd.Series,
    *,
    train_year_end: int,
    test_year_start: int,
    test_year_end: int,
) -> tuple[pd.Series, pd.Series]:
    train_mask = years <= train_year_end
    test_mask = (years >= test_year_start) & (years <= test_year_end)
    return train_mask, test_mask


def split_train_test_temporal(df: pd.DataFrame, cfg: ScreenOracleConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask, test_mask = temporal_masks(
        df["startYear"],
        train_year_end=cfg.train_year_end,
        test_year_start=cfg.test_year_start,
        test_year_end=cfg.test_year_end,
    )
    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()
    return train_df, test_df


def prepare_modeling_tables(
    basics: pd.DataFrame,
    ratings: pd.DataFrame,
    cfg: ScreenOracleConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build cleaned merged table, then train/test split with genre collapsing fit on train.

    Returns (train_df, test_df, full_df) where full_df is merged+cleaned before genre collapse.
    """
    merged = merge_basics_and_ratings(basics, ratings)
    merged = keep_movies_only(merged)
    merged = coerce_core_types(merged)
    merged = add_success_label(
        merged,
        min_votes=cfg.min_votes_for_label,
        rating_threshold=cfg.rating_threshold,
    )
    merged = drop_unusable_rows_for_features(merged)
    merged["primary_genre"] = extract_primary_genre(merged["genres"])

    train_df, test_df = split_train_test_temporal(merged, cfg)
    allow = fit_top_genre_allowlist(train_df["primary_genre"], top_k=cfg.top_genres)
    train_df["primary_genre"] = collapse_rare_genres(train_df["primary_genre"], allow)
    test_df["primary_genre"] = collapse_rare_genres(test_df["primary_genre"], allow)
    return train_df, test_df, merged


def xy_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = ["startYear", "runtimeMinutes", "isAdult", "primary_genre"]
    x = df[feature_cols].copy()
    y = df["success"].copy()
    return x, y


def training_class_balance_report(y: pd.Series) -> dict[str, float]:
    vals, counts = np.unique(y.to_numpy(), return_counts=True)
    total = counts.sum()
    out: dict[str, float] = {}
    for v, c in zip(vals, counts, strict=True):
        out[f"class_{int(v)}"] = float(c)
        out[f"class_{int(v)}_share"] = float(c) / float(total)
    return out

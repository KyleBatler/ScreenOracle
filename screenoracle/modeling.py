from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from screenoracle.config import ScreenOracleConfig


def build_model_pipeline(cfg: ScreenOracleConfig) -> Pipeline:
    """
    Training pipeline:

    - Numeric: median imputation for missing runtime
    - Categorical: one-hot genres with unknown handling
    - Model: random forest (strong baseline on mixed data)
    """
    numeric_features = ["startYear", "runtimeMinutes", "isAdult"]
    categorical_features = ["primary_genre"]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=cfg.model_n_estimators,
        max_depth=cfg.model_max_depth,
        min_samples_leaf=cfg.model_min_samples_leaf,
        random_state=cfg.random_state,
        n_jobs=cfg.model_n_jobs,
        class_weight="balanced_subsample",
    )
    return Pipeline(steps=[("prep", pre), ("model", model)])


def fit_pipeline(pipe: Pipeline, x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipe.fit(x_train, y_train)
    return pipe


def tune_threshold_f1(y_true: pd.Series, proba: np.ndarray) -> float:
    """
    Pick a decision threshold by maximizing F1 on a validation set.

    This helps when classes are imbalanced and 0.5 is not a reasonable cut point.
    """
    y = y_true.to_numpy(dtype=int)
    best_t = 0.5
    best_f1 = -1.0
    for t in np.arange(0.02, 0.99, 0.02):
        pred = (proba >= t).astype(int)
        score = f1_score(y, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t


def evaluate_pipeline(
    pipe: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    threshold: float = 0.5,
) -> dict[str, object]:
    y_true = y_test.to_numpy(dtype=int)
    proba = pipe.predict_proba(x_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    metrics: dict[str, object] = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "threshold": float(threshold),
        "positive_rate_true": float(np.mean(y_true == 1)),
        "positive_rate_pred": float(np.mean(pred == 1)),
        "accuracy": float(np.mean(pred == y_true)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def save_pipeline(pipe: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)


def load_pipeline(path: Path) -> Pipeline:
    obj = joblib.load(path)
    if not isinstance(obj, Pipeline):
        raise TypeError(f"Expected sklearn Pipeline at {path}, got {type(obj)!r}")
    return obj


def predict_success_proba(pipe: Pipeline, row: dict[str, object]) -> float:
    x = pd.DataFrame([row])
    return float(pipe.predict_proba(x)[0, 1])


def forest_feature_importances(pipe: Pipeline) -> pd.Series | None:
    """Return importances for tree models; None if not available."""
    model = pipe.named_steps.get("model")
    if not hasattr(model, "feature_importances_"):
        return None
    prep: ColumnTransformer = pipe.named_steps["prep"]
    feature_names = prep.get_feature_names_out()
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

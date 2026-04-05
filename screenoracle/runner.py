from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from screenoracle.config import DEFAULT_CONFIG, ScreenOracleConfig
from screenoracle.dataset import (
    prepare_modeling_tables,
    training_class_balance_report,
    xy_split,
)
from screenoracle.i18n import Lang, pct_from_share, pretty_feature_name, t
from screenoracle.imdb_io import (
    assert_files_non_empty,
    get_default_imdb_paths,
    load_title_basics,
    load_title_ratings,
)
from screenoracle.modeling import (
    build_model_pipeline,
    evaluate_pipeline,
    fit_pipeline,
    forest_feature_importances,
    load_pipeline,
    save_pipeline,
    tune_threshold_f1,
)
from screenoracle.paths import ensure_dir, get_project_root


def print_header(title: str) -> None:
    print()
    n = max(len(title), 3)
    print("=" * n)
    print(title)
    print("=" * n)


def run_train_eval_and_save(cfg: ScreenOracleConfig | None = None, *, lang: Lang) -> Path:
    cfg = cfg or DEFAULT_CONFIG
    root = get_project_root()

    basics_path, ratings_path = get_default_imdb_paths(
        imdb_subdir=cfg.imdb_subdir,
        basics_name=cfg.basics_name,
        ratings_name=cfg.ratings_name,
        project_root=root,
    )
    assert_files_non_empty([basics_path, ratings_path], lang=lang)

    print_header(t(lang, "step_load"))
    print(t(lang, "file_movies"), basics_path)
    print(t(lang, "file_scores"), ratings_path)

    basics = load_title_basics(basics_path)
    ratings = load_title_ratings(ratings_path)

    dup_b = int(basics["tconst"].duplicated().sum())
    dup_r = int(ratings["tconst"].duplicated().sum())
    if dup_b or dup_r:
        print(t(lang, "warn_dupes", dup_b=dup_b, dup_r=dup_r))

    print_header(t(lang, "step_build"))
    train_df, test_df, _full = prepare_modeling_tables(basics, ratings, cfg)
    print(
        t(
            lang,
            "rows_learn",
            n=len(train_df),
            end=cfg.train_year_end,
        )
    )
    print(
        t(
            lang,
            "rows_check",
            n=len(test_df),
            a=cfg.test_year_start,
            b=cfg.test_year_end,
        )
    )

    x_train, y_train = xy_split(train_df)
    x_test, y_test = xy_split(test_df)

    train_bal = training_class_balance_report(y_train)
    test_bal = training_class_balance_report(y_test)
    train_pct = pct_from_share(train_bal.get("class_1_share", 0.0))
    test_pct = pct_from_share(test_bal.get("class_1_share", 0.0))
    print(t(lang, "share_train", pct=train_pct))
    print(t(lang, "share_test", pct=test_pct))

    print_header(t(lang, "step_train"))
    x_fit, x_cal, y_fit, y_cal = train_test_split(
        x_train,
        y_train,
        test_size=0.15,
        stratify=y_train,
        random_state=cfg.random_state,
    )
    pipe = build_model_pipeline(cfg)
    fit_pipeline(pipe, x_fit, y_fit)
    proba_cal = pipe.predict_proba(x_cal)[:, 1]
    threshold = tune_threshold_f1(y_cal, proba_cal)
    print(t(lang, "chosen_threshold", t=threshold))
    fit_pipeline(pipe, x_train, y_train)

    print_header(t(lang, "step_test"))
    metrics = evaluate_pipeline(pipe, x_test, y_test, threshold=threshold)
    print(t(lang, "score_quality", auc=metrics["roc_auc"]))
    print(t(lang, "line_threshold", t=metrics["threshold"]))
    print(
        t(
            lang,
            "line_true_success_share",
            pct=pct_from_share(metrics["positive_rate_true"]),
        )
    )
    print(
        t(
            lang,
            "line_pred_success_share",
            pct=pct_from_share(metrics["positive_rate_pred"]),
        )
    )
    print(
        t(
            lang,
            "line_accuracy",
            pct=pct_from_share(metrics["accuracy"]),
        )
    )
    print(t(lang, "confusion_title"))
    print(t(lang, "cm_tn", n=metrics["tn"]))
    print(t(lang, "cm_fp", n=metrics["fp"]))
    print(t(lang, "cm_fn", n=metrics["fn"]))
    print(t(lang, "cm_tp", n=metrics["tp"]))

    imp = forest_feature_importances(pipe)
    if imp is not None:
        print_header(t(lang, "feat_title"))
        for feat, score in imp.head(15).items():
            label = pretty_feature_name(lang, str(feat))
            print(f"  {label}: {float(score):.3f}")

    models_dir = root / cfg.models_dir
    ensure_dir(models_dir)
    out_path = models_dir / cfg.artifact_name
    save_pipeline(pipe, out_path)
    print_header(t(lang, "saved_title"))
    print(str(out_path.resolve()))

    thr_path = models_dir / cfg.threshold_name
    thr_path.write_text(f"{threshold:.6f}\n", encoding="utf-8")
    print(t(lang, "saved_threshold", path=str(thr_path.resolve()), t=threshold))

    meta_path = models_dir / "last_run_config.txt"
    lines = [f"{k}={v}" for k, v in asdict(cfg).items()]
    meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(t(lang, "saved_config", path=str(meta_path.resolve())))

    return out_path


def run_predict_demo(cfg: ScreenOracleConfig | None = None, *, lang: Lang, n_samples: int = 8) -> None:
    cfg = cfg or DEFAULT_CONFIG
    root = get_project_root()
    model_path = root / cfg.models_dir / cfg.artifact_name
    thr_path = root / cfg.models_dir / cfg.threshold_name
    if not model_path.is_file():
        raise FileNotFoundError(t(lang, "err_no_model", path=str(model_path.resolve())))

    basics_path, ratings_path = get_default_imdb_paths(
        imdb_subdir=cfg.imdb_subdir,
        basics_name=cfg.basics_name,
        ratings_name=cfg.ratings_name,
        project_root=root,
    )
    assert_files_non_empty([basics_path, ratings_path], lang=lang)

    basics = load_title_basics(basics_path)
    ratings = load_title_ratings(ratings_path)
    train_df, test_df, _merged = prepare_modeling_tables(basics, ratings, cfg)

    if thr_path.is_file():
        threshold = float(thr_path.read_text(encoding="utf-8").strip())
    else:
        _x_train, y_train = xy_split(train_df)
        x_fit, x_cal, y_fit, y_cal = train_test_split(
            _x_train,
            y_train,
            test_size=0.15,
            stratify=y_train,
            random_state=cfg.random_state,
        )
        pipe_cal = build_model_pipeline(cfg)
        fit_pipeline(pipe_cal, x_fit, y_fit)
        proba_cal = pipe_cal.predict_proba(x_cal)[:, 1]
        threshold = tune_threshold_f1(y_cal, proba_cal)
        word = "Внимание" if lang == "ru" else "Warning"
        print(f"{word}: no {cfg.threshold_name}; recomputed threshold={threshold:.4f}")

    pipe = load_pipeline(model_path)
    sample = test_df.sample(n=min(n_samples, len(test_df)), random_state=cfg.random_state)
    xs, _ys = xy_split(sample)

    print_header(t(lang, "demo_title"))
    print(t(lang, "demo_model"), model_path.resolve())
    print(t(lang, "demo_threshold"), f"{threshold:.2f}")
    print()

    proba = pipe.predict_proba(xs)[:, 1]
    pred = (proba >= threshold).astype(int)
    for idx, row in sample.reset_index(drop=True).iterrows():
        title = str(row.get("primaryTitle", ""))
        raw_year = row.get("startYear", "")
        try:
            year = int(float(raw_year))
        except (TypeError, ValueError):
            year = raw_year
        genre = str(row.get("primary_genre", ""))
        truth = int(row["success"])
        pred_word = t(lang, "demo_pred_yes") if pred[idx] == 1 else t(lang, "demo_pred_no")
        truth_word = t(lang, "demo_truth_yes") if truth == 1 else t(lang, "demo_truth_no")
        print(
            t(
                lang,
                "demo_row",
                title=title,
                year=year,
                genre=genre,
                p=float(proba[idx]),
                pred=pred_word,
                truth=truth_word,
            )
        )

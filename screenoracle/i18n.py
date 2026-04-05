from __future__ import annotations

from typing import Literal

Lang = Literal["ru", "en"]

_TEXT: dict[str, dict[str, str]] = {
    "ru": {
        "app_title": "ScreenOracle",
        "step_load": "Шаг 1. Загружаем данные IMDb",
        "file_movies": "Описание фильмов:",
        "file_scores": "Оценки и число голосов:",
        "warn_dupes": "Внимание: есть повторяющиеся id — лишние строки отброшены. basics={dup_b}, ratings={dup_r}",
        "step_build": "Шаг 2. Собираем таблицу: только фильмы, метка «успех», делим старые / новее годы",
        "rows_learn": "Фильмов для обучения (годы до {end} включительно): {n:,}",
        "rows_check": "Фильмов для проверки (годы с {a} по {b}): {n:,}",
        "share_train": "В обучении — «успех»: {pct:.1f}% строк (остальное — неуспех по нашему правилу).",
        "share_test": "В проверке — «успех»: {pct:.1f}% строк.",
        "step_train": "Шаг 3. Обучение модели и подбор порога «да/нет» (только на части обучающих данных)",
        "chosen_threshold": "Выбранный порог вероятности (выше — считаем «успех»): {t:.2f}",
        "step_test": "Шаг 4. Проверка на более новых годах (модель их не видела при обучении)",
        "score_quality": "Насколько модель умеет отличать «успех» от «неуспеха» (от 0.5 до 1; чем ближе к 1 — тем лучше): {auc:.2f}",
        "line_threshold": "Порог: {t:.2f}",
        "line_true_success_share": "Доля настоящих «успехов» среди проверочных фильмов: {pct:.1f}%",
        "line_pred_success_share": "Долю фильмов модель назвала «успехом»: {pct:.1f}%",
        "line_accuracy": "Доля угаданных ответов среди всех проверочных фильмов: {pct:.1f}%",
        "confusion_title": "Разбор ответов на проверке (простыми словами):",
        "cm_tn": "  Правильно сказали «неуспех»: {n:,}",
        "cm_fp": "  Ошибка: сказали «успех», а по факту нет: {n:,}",
        "cm_fn": "  Ошибка: сказали «неуспех», а по факту успех: {n:,}",
        "cm_tp": "  Правильно сказали «успех»: {n:,}",
        "feat_title": "Шаг 5. На что модель опиралась сильнее всего (очень грубо, не «причины» в жизни)",
        "saved_title": "Готово: модель сохранена",
        "saved_threshold": "Порог сохранён: {path} (значение {t:.4f})",
        "saved_config": "Настройки прогона сохранены: {path}",
        "demo_title": "Пример: несколько фильмов из «нового» периода",
        "demo_model": "Файл модели:",
        "demo_threshold": "Порог:",
        "demo_row": "«{title}» ({year}) | жанр: {genre} | шанс «успеха»: {p:.0%} | ответ модели: {pred} | по факту: {truth}",
        "demo_pred_yes": "успех",
        "demo_pred_no": "неуспех",
        "demo_truth_yes": "успех",
        "demo_truth_no": "неуспех",
        "err_no_model": "Модель не найдена: {path}\nСначала выполни: python run_screenoracle.py --lang ru train",
        "err_missing_file": "Не хватает файлов IMDb:\n{paths}\nПоложи скачанные .tsv.gz в папку data/raw/imdb/",
    },
    "en": {
        "app_title": "ScreenOracle",
        "step_load": "Step 1. Loading IMDb files",
        "file_movies": "Movie info file:",
        "file_scores": "Ratings / vote counts file:",
        "warn_dupes": "Warning: duplicate ids found — extras dropped. basics={dup_b}, ratings={dup_r}",
        "step_build": "Step 2. Build table: movies only, success label, split older / newer years",
        "rows_learn": "Movies for training (years up to {end}): {n:,}",
        "rows_check": "Movies for checking (years {a}–{b}): {n:,}",
        "share_train": "In training, “success” rows: {pct:.1f}% (rest = not success by our rule).",
        "share_test": "In the check set, “success” rows: {pct:.1f}%.",
        "step_train": "Step 3. Train model and pick yes/no cutoff (using only part of training data)",
        "chosen_threshold": "Probability cutoff (above = we call it “success”): {t:.2f}",
        "step_test": "Step 4. Check on newer years (the model never saw these rows while training)",
        "score_quality": "How well the model separates success vs not (0.5–1 scale; higher is better): {auc:.2f}",
        "line_threshold": "Cutoff: {t:.2f}",
        "line_true_success_share": "Share of real successes in the check set: {pct:.1f}%",
        "line_pred_success_share": "Share the model called “success”: {pct:.1f}%",
        "line_accuracy": "Share of correct answers on all check movies: {pct:.1f}%",
        "confusion_title": "What happened on the check set (plain words):",
        "cm_tn": "  Correct “not success”: {n:,}",
        "cm_fp": "  Wrong: said “success”, but it was not: {n:,}",
        "cm_fn": "  Wrong: said “not success”, but it was a success: {n:,}",
        "cm_tp": "  Correct “success”: {n:,}",
        "feat_title": "Step 5. What the model leaned on most (very rough, not real-world “causes”)",
        "saved_title": "Done: model saved",
        "saved_threshold": "Cutoff saved: {path} (value {t:.4f})",
        "saved_config": "Run settings saved: {path}",
        "demo_title": "Demo: a few movies from the “newer” period",
        "demo_model": "Model file:",
        "demo_threshold": "Cutoff:",
        "demo_row": "“{title}” ({year}) | genre: {genre} | chance of “success”: {p:.0%} | model says: {pred} | really: {truth}",
        "demo_pred_yes": "success",
        "demo_pred_no": "not success",
        "demo_truth_yes": "success",
        "demo_truth_no": "not success",
        "err_no_model": "Model not found: {path}\nFirst run: python run_screenoracle.py --lang en train",
        "err_missing_file": "Missing IMDb files:\n{paths}\nPut downloaded .tsv.gz files in data/raw/imdb/",
    },
}


def t(lang: Lang, key: str, **kwargs: object) -> str:
    block = _TEXT.get(lang) or _TEXT["en"]
    template = block.get(key) or _TEXT["en"][key]
    return template.format(**kwargs)


def pct_from_share(share: float) -> float:
    return float(share) * 100.0


def pretty_feature_name(lang: Lang, raw: str) -> str:
    if raw.startswith("num__"):
        key = raw[len("num__") :]
        if lang == "ru":
            names = {
                "runtimeMinutes": "длительность (минуты)",
                "startYear": "год выхода",
                "isAdult": "пометка взрослого контента",
            }
        else:
            names = {
                "runtimeMinutes": "runtime (minutes)",
                "startYear": "release year",
                "isAdult": "adult title flag",
            }
        return names.get(key, key)
    if raw.startswith("cat__primary_genre_"):
        g = raw[len("cat__primary_genre_") :]
        return f"жанр: {g}" if lang == "ru" else f"genre: {g}"
    return raw

#!/usr/bin/env python3
"""Train/evaluate ScreenOracle on local IMDb TSV.GZ dumps."""

from __future__ import annotations

import argparse

from screenoracle.i18n import Lang
from screenoracle.runner import run_predict_demo, run_train_eval_and_save


def main() -> None:
    p = argparse.ArgumentParser(description="ScreenOracle — IMDb movie success baseline")
    p.add_argument(
        "--lang",
        choices=["ru", "en"],
        required=True,
        help="Output language: ru (Russian) or en (English)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train", help="Load IMDb, train model, check on newer years, save files")

    d = sub.add_parser("demo", help="Show a few examples (run train first)")
    d.add_argument("--n", type=int, default=8, help="How many random rows to show")

    args = p.parse_args()
    lang: Lang = args.lang  # type: ignore[assignment]

    if args.cmd == "train":
        run_train_eval_and_save(lang=lang)
    elif args.cmd == "demo":
        run_predict_demo(lang=lang, n_samples=args.n)


if __name__ == "__main__":
    main()

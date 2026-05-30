#!/usr/bin/env python3
"""
Plot per-dataset prediction timing curves with 95% confidence intervals.

This is based on:
  /home/kerith/workspaces/catboost/test/plot_prediction_timing_benchmark.py

Input:
  /home/kerith/workspaces/smooth_multi/prediction_duration_benchmark.csv

Outputs:
  - one PNG chart per dataset
  - one summary CSV with means / SEM / 95% CI per dataset-depth-series
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = Path("/home/kerith/workspaces/smooth_multi/prediction_duration_benchmark.csv")
OUTPUT_DIR = Path("/home/kerith/workspaces/smooth_multi/prediction_duration_charts")
SUMMARY_PATH = OUTPUT_DIR / "prediction_duration_benchmark_summary.csv"

SERIES_ORDER = [
    "catboost_unsmoothed",
    "catboost_smoothed",
    "xgboost",
    "lightgbm",
]

SERIES_LABELS = {
    "catboost_unsmoothed": "CatBoost, interpolation disabled",
    "catboost_smoothed": "CatBoost, interpolation enabled",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

SERIES_COLORS = {
    "catboost_unsmoothed": "#4C78A8",
    "catboost_smoothed": "#F58518",
    "xgboost": "#54A24B",
    "lightgbm": "#B279A2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot prediction duration benchmark charts."
    )
    parser.add_argument(
        "--only-dataset",
        default=None,
        help="Only plot this exact dataset name.",
    )
    parser.add_argument(
        "--only-dataset-name",
        default=None,
        help="Only plot datasets whose name contains this case-insensitive text.",
    )
    return parser.parse_args()


def dataset_title(name: str) -> str:
    return name.replace("_", " ").title()


def load_benchmark() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)

    catboost_mask = df["model_type"] == "catboost"
    df.loc[catboost_mask & (df["interpolation_enabled"] == False), "series"] = "catboost_unsmoothed"
    df.loc[catboost_mask & (df["interpolation_enabled"] == True), "series"] = "catboost_smoothed"
    df.loc[df["model_type"] == "xgboost", "series"] = "xgboost"
    df.loc[df["model_type"] == "lightgbm", "series"] = "lightgbm"

    df["prediction_duration_ms"] = df["prediction_duration_seconds"] * 1000.0
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["dataset", "depth", "series"], as_index=False)
        .agg(
            mean_prediction_duration_ms=("prediction_duration_ms", "mean"),
            sem_prediction_duration_ms=("prediction_duration_ms", "sem"),
            n_models=("prediction_duration_ms", "size"),
        )
        .sort_values(["dataset", "series", "depth"])
    )
    summary["ci95_prediction_duration_ms"] = 1.96 * summary["sem_prediction_duration_ms"]
    return summary


def plot_dataset(summary: pd.DataFrame, dataset: str) -> Path:
    dataset_summary = summary[summary["dataset"] == dataset]
    if dataset_summary.empty:
        raise ValueError(f"No benchmark rows found for dataset={dataset}")

    fig, ax = plt.subplots(figsize=(8.5, 5.25), constrained_layout=True)

    for series_name in SERIES_ORDER:
        series = dataset_summary[dataset_summary["series"] == series_name].sort_values("depth")
        if series.empty:
            continue

        ax.errorbar(
            series["depth"],
            series["mean_prediction_duration_ms"],
            yerr=series["ci95_prediction_duration_ms"],
            marker="o",
            linewidth=2,
            capsize=4,
            label=SERIES_LABELS[series_name],
            color=SERIES_COLORS[series_name],
        )

    ax.set_title(f"{dataset_title(dataset)} Prediction Duration by Depth")
    ax.set_xlabel("Model depth")
    ax.set_ylabel("Prediction duration (ms)")
    ax.set_xticks(sorted(dataset_summary["depth"].unique()))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    output_path = OUTPUT_DIR / f"{dataset}_prediction_duration_by_depth.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def filter_datasets(datasets: list[str], args: argparse.Namespace) -> list[str]:
    if args.only_dataset:
        datasets = [dataset for dataset in datasets if dataset == args.only_dataset]
    if args.only_dataset_name:
        needle = args.only_dataset_name.lower()
        datasets = [dataset for dataset in datasets if needle in dataset.lower()]
    return datasets


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary(load_benchmark())
    datasets = filter_datasets(sorted(summary["dataset"].unique()), args)
    if not datasets:
        print("No datasets matched the requested filters.")
        return

    summary = summary[summary["dataset"].isin(datasets)].reset_index(drop=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    output_paths = []
    for dataset in datasets:
        output_paths.append(plot_dataset(summary, dataset))

    print(f"Saved summary: {SUMMARY_PATH}")
    for output_path in output_paths:
        print(f"Saved chart: {output_path}")


if __name__ == "__main__":
    main()

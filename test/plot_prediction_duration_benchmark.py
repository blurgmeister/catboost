#!/usr/bin/env python3
"""
Plot per-dataset prediction timing curves with 95% confidence intervals.

This is based on:
  /home/kerith/workspaces/catboost/test/plot_prediction_timing_benchmark.py

Input:
  /home/kerith/workspaces/smooth_multi/prediction_duration_benchmark.csv

Outputs:
  - one PNG chart per dataset
  - one PNG overview showing the slowest series per dataset-depth
  - one PNG bar chart counting slowest-model occurrences by depth
  - one summary CSV with means / SEM / 95% CI per dataset-depth-series
  - one high-level CSV with the slowest series per dataset-depth
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.stats import t as student_t


INPUT_PATH = Path("/home/kerith/workspaces/smooth_multi/prediction_duration_benchmark.csv")
OUTPUT_DIR = Path("/home/kerith/workspaces/smooth_multi/prediction_duration_charts")
SUMMARY_PATH = OUTPUT_DIR / "prediction_duration_benchmark_summary.csv"
HIGHLEVEL_SUMMARY_PATH = OUTPUT_DIR / "prediction_duration_benchmark_highlevel.csv"
SLOWEST_PLOT_PATH = OUTPUT_DIR / "prediction_duration_slowest.png"
SLOWEST_BARCHART_PATH = OUTPUT_DIR / "prediction_duration_barchart.png"
TWO_SIDED_95_PERCENT_QUANTILE = 0.975

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
    degrees_of_freedom = summary["n_models"] - 1
    critical_values = student_t.ppf(
        TWO_SIDED_95_PERCENT_QUANTILE,
        degrees_of_freedom,
    )
    summary["ci95_prediction_duration_ms"] = (
        critical_values * summary["sem_prediction_duration_ms"]
    )
    return summary


def build_highlevel_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.sort_values(
            ["dataset", "depth", "mean_prediction_duration_ms", "series"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["dataset", "depth"], keep="first")
        .reset_index(drop=True)
    )


def plot_slowest_series(highlevel_summary: pd.DataFrame) -> Path:
    datasets = sorted(highlevel_summary["dataset"].unique())
    dataset_positions = {dataset: position for position, dataset in enumerate(datasets)}
    figure_height = max(5.25, 0.4 * len(datasets) + 1.5)
    fig, ax = plt.subplots(figsize=(9, figure_height), constrained_layout=True)

    for series_name in SERIES_ORDER:
        series = highlevel_summary[highlevel_summary["series"] == series_name]
        if series.empty:
            continue

        ax.scatter(
            series["depth"],
            series["dataset"].map(dataset_positions),
            s=70,
            color=SERIES_COLORS[series_name],
            label=SERIES_LABELS[series_name],
            zorder=3,
        )

    ax.set_title("Slowest Prediction Model by Dataset and Depth")
    ax.set_xlabel("Model depth")
    ax.set_ylabel("Dataset")
    ax.set_xticks(sorted(highlevel_summary["depth"].unique()))
    ax.set_yticks(range(len(datasets)), [dataset_title(dataset) for dataset in datasets])
    ax.invert_yaxis()
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(
        title="Model",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
    )

    fig.savefig(SLOWEST_PLOT_PATH, dpi=200)
    plt.close(fig)
    return SLOWEST_PLOT_PATH


def plot_slowest_model_counts(highlevel_summary: pd.DataFrame) -> Path:
    depths = sorted(highlevel_summary["depth"].unique())
    counts = (
        highlevel_summary.groupby(["depth", "series"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=depths, columns=SERIES_ORDER, fill_value=0)
    )

    x_positions = np.arange(len(depths))
    cumulative_counts = np.zeros(len(depths))
    fig, ax = plt.subplots(figsize=(9, 5.25), constrained_layout=True)

    for series_name in SERIES_ORDER:
        ax.bar(
            x_positions,
            counts[series_name],
            width=0.7,
            bottom=cumulative_counts,
            color=SERIES_COLORS[series_name],
            label=SERIES_LABELS[series_name],
        )
        cumulative_counts += counts[series_name].to_numpy()

    ax.set_title("Slowest Model Counts by Depth")
    ax.set_xlabel("Model depth")
    ax.set_ylabel("Number of datasets where slowest")
    ax.set_xticks(x_positions, depths)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        title="Model",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
    )

    fig.savefig(SLOWEST_BARCHART_PATH, dpi=200)
    plt.close(fig)
    return SLOWEST_BARCHART_PATH


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
    highlevel_summary = build_highlevel_summary(summary)
    highlevel_summary.to_csv(HIGHLEVEL_SUMMARY_PATH, index=False)
    slowest_plot_path = plot_slowest_series(highlevel_summary)
    slowest_barchart_path = plot_slowest_model_counts(highlevel_summary)

    output_paths = []
    for dataset in datasets:
        output_paths.append(plot_dataset(summary, dataset))

    print(f"Saved summary: {SUMMARY_PATH}")
    print(f"Saved high-level summary: {HIGHLEVEL_SUMMARY_PATH}")
    print(f"Saved slowest-series chart: {slowest_plot_path}")
    print(f"Saved slowest-model bar chart: {slowest_barchart_path}")
    for output_path in output_paths:
        print(f"Saved chart: {output_path}")


if __name__ == "__main__":
    main()

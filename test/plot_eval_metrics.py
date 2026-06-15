#!/usr/bin/env python3
"""Plot train and validation evaluation metrics from the duration benchmark."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT_PATH = Path(
    "/home/kerith/workspaces/smooth_multi/prediction_duration_benchmark.csv"
)
DEFAULT_OUTPUT_DIR = Path("/home/kerith/workspaces/smooth_multi/eval_metrics")

METRICS = (
    ("train_eval_metric", "Train evaluation metric"),
    ("validation_eval_metric", "Validation evaluation metric"),
)

SERIES_ORDER = (
    "catboost_unsmoothed",
    "catboost_smoothed",
    "xgboost",
    "lightgbm",
)

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

SERIES_STYLES = {
    "catboost_unsmoothed": {
        "linestyle": "-",
        "marker": "o",
        "markerfacecolor": "#4C78A8",
        "alpha": 0.9,
        "zorder": 3,
    },
    "catboost_smoothed": {
        "linestyle": "--",
        "marker": "D",
        "markerfacecolor": "none",
        "alpha": 0.85,
        "zorder": 4,
    },
    "xgboost": {
        "linestyle": "-",
        "marker": "s",
        "markerfacecolor": "#54A24B",
        "alpha": 0.9,
        "zorder": 2,
    },
    "lightgbm": {
        "linestyle": "-",
        "marker": "^",
        "markerfacecolor": "#B279A2",
        "alpha": 0.9,
        "zorder": 1,
    },
}

OUTLIER_SCALE_RATIO = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot benchmark evaluation metric means with 95% SEM bars."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--only-dataset",
        help="Only plot the dataset with this exact name.",
    )
    return parser.parse_args()


def add_series_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    catboost = df["model_type"].eq("catboost")
    interpolation_enabled = df["interpolation_enabled"].astype("boolean")

    df.loc[catboost & interpolation_enabled.eq(False), "series"] = (
        "catboost_unsmoothed"
    )
    df.loc[catboost & interpolation_enabled.eq(True), "series"] = (
        "catboost_smoothed"
    )
    df.loc[df["model_type"].eq("xgboost"), "series"] = "xgboost"
    df.loc[df["model_type"].eq("lightgbm"), "series"] = "lightgbm"
    return df


def load_benchmark(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    required_columns = {
        "dataset",
        "depth",
        "model_type",
        "interpolation_enabled",
        "train_eval_metric",
        "validation_eval_metric",
    }
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = add_series_column(df)
    unknown_models = df[df["series"].isna()]["model_type"].dropna().unique()
    if len(unknown_models):
        print(f"Skipping unsupported model types: {', '.join(sorted(unknown_models))}")
    return df[df["series"].notna()].copy()


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["dataset", "depth", "series"], as_index=False)
        .agg(
            mean_train_eval_metric=("train_eval_metric", "mean"),
            sem_train_eval_metric=("train_eval_metric", "sem"),
            mean_validation_eval_metric=("validation_eval_metric", "mean"),
            sem_validation_eval_metric=("validation_eval_metric", "sem"),
            n_models=("train_eval_metric", "size"),
        )
        .sort_values(["dataset", "series", "depth"])
    )
    summary["ci95_train_eval_metric"] = 1.96 * summary[
        "sem_train_eval_metric"
    ].fillna(0.0)
    summary["ci95_validation_eval_metric"] = 1.96 * summary[
        "sem_validation_eval_metric"
    ].fillna(0.0)
    return summary


def dataset_title(dataset: str) -> str:
    return dataset.replace("_", " ").replace("-", " ").title()


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def limit_axis_for_outliers(
    ax: plt.Axes,
    dataset_summary: pd.DataFrame,
    metric: str,
) -> None:
    mean_column = f"mean_{metric}"
    ci_column = f"ci95_{metric}"
    series_maxima = dataset_summary.groupby("series")[mean_column].max().sort_values()
    if len(series_maxima) < 2:
        return

    largest = series_maxima.iloc[-1]
    second_largest = series_maxima.iloc[-2]
    if second_largest <= 0 or largest <= OUTLIER_SCALE_RATIO * second_largest:
        return

    outlier_series = series_maxima.index[-1]
    visible = dataset_summary[~dataset_summary["series"].eq(outlier_series)]
    visible_lower = (visible[mean_column] - visible[ci_column]).min()
    visible_upper = (visible[mean_column] + visible[ci_column]).max()
    padding = max((visible_upper - visible_lower) * 0.08, abs(visible_upper) * 0.02)
    ax.set_ylim(visible_lower - padding, visible_upper + padding)
    ax.text(
        0.02,
        0.98,
        f"{SERIES_LABELS[outlier_series]} values exceed axis limit",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        color=SERIES_COLORS[outlier_series],
    )


def plot_dataset(summary: pd.DataFrame, dataset: str, output_dir: Path) -> Path:
    dataset_summary = summary[summary["dataset"].eq(dataset)]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.25),
        sharex=True,
        constrained_layout=True,
    )

    for ax, (metric, title) in zip(axes, METRICS):
        for series_name in SERIES_ORDER:
            series = dataset_summary[
                dataset_summary["series"].eq(series_name)
            ].sort_values("depth")
            if series.empty:
                continue

            ax.errorbar(
                series["depth"],
                series[f"mean_{metric}"],
                yerr=series[f"ci95_{metric}"],
                linewidth=2,
                capsize=4,
                color=SERIES_COLORS[series_name],
                label=SERIES_LABELS[series_name],
                markeredgecolor=SERIES_COLORS[series_name],
                markeredgewidth=1.5,
                markersize=6,
                **SERIES_STYLES[series_name],
            )

        ax.set_title(title)
        ax.set_xlabel("Model depth")
        ax.set_ylabel("Evaluation metric")
        ax.set_xticks(sorted(dataset_summary["depth"].unique()))
        ax.grid(axis="y", alpha=0.25)
        limit_axis_for_outliers(ax, dataset_summary, metric)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(f"{dataset_title(dataset)} Evaluation Metrics")

    output_path = output_dir / f"{safe_filename(dataset)}_eval_metrics.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    df = load_benchmark(args.input)
    if args.only_dataset:
        df = df[df["dataset"].eq(args.only_dataset)]
    if df.empty:
        raise ValueError("No benchmark rows matched the requested dataset selection.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(df)
    summary_path = args.output_dir / "eval_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)

    output_paths = [
        plot_dataset(summary, dataset, args.output_dir)
        for dataset in sorted(summary["dataset"].unique())
    ]

    print(f"Saved summary: {summary_path}")
    for output_path in output_paths:
        print(f"Saved chart: {output_path}")


if __name__ == "__main__":
    main()

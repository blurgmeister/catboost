#!/usr/bin/env python3
"""Plot seed-averaged decile charts with 95% confidence intervals."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = Path("/home/kerith/workspaces/smooth_multi/decile/decile_summary.csv")
OUTPUT_DIR = Path("/home/kerith/workspaces/smooth_multi/decile")
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "decile_summary_small.csv"
EXPECTED_SEED_COUNT = 50
NORMAL_95_PERCENT_CRITICAL_VALUE = 1.959963984540054
GROUP_COLUMNS = ["dataset", "depth", "class", "decile_bucket"]
MEASURES = (
    ("average_actuals", "Actuals", "#4C78A8", "o", "-"),
    (
        "average_unsmoothed_predictions",
        "Unsmoothed predictions",
        "#F58518",
        "s",
        "--",
    ),
    (
        "average_smoothed_predictions",
        "Smoothed predictions",
        "#54A24B",
        "D",
        "-.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot decile means and 95% confidence intervals across random seeds."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=INPUT_PATH,
        help="Seed-level decile summary CSV produced by compute_deciles.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to receive the decile chart PNG files.",
    )
    parser.add_argument(
        "--summary-output-path",
        type=Path,
        default=SUMMARY_OUTPUT_PATH,
        help="CSV file to receive the seed-aggregated chart datapoints.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Resolution of the output PNG files.",
    )
    return parser.parse_args()


def safe_filename_part(value: object) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    result = re.sub(r"_+", "_", result).strip("_.")
    return result[:120] or "value"


def display_dataset_name(dataset: str) -> str:
    return dataset.replace("_", " ")


def load_deciles(input_path: Path) -> pd.DataFrame:
    if not input_path.is_file():
        raise FileNotFoundError(f"Decile summary does not exist: {input_path}")

    frame = pd.read_csv(input_path, keep_default_na=False)
    required_columns = {
        "dataset",
        "depth",
        "random_seed",
        "class",
        "decile_bucket",
        *(measure for measure, *_ in MEASURES),
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Decile summary is missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Decile summary contains no rows")

    frame = frame.copy()
    frame["dataset"] = frame["dataset"].astype(str)
    frame["class"] = frame["class"].astype(str)
    for column in ("depth", "random_seed", "decile_bucket"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    for measure, *_ in MEASURES:
        frame[measure] = pd.to_numeric(frame[measure], errors="coerce")

    invalid_buckets = frame.loc[~frame["decile_bucket"].between(1, 10), "decile_bucket"]
    if not invalid_buckets.empty:
        raise ValueError("decile_bucket values must be integers from 1 through 10")

    observation_key = ["dataset", "depth", "class", "decile_bucket", "random_seed"]
    duplicates = frame.duplicated(observation_key, keep=False)
    if duplicates.any():
        example = frame.loc[duplicates, observation_key].iloc[0].to_dict()
        raise ValueError(f"Duplicate seed-level decile row found: {example}")
    return frame


def summarize_across_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "seed_count": ("random_seed", "nunique")
    }
    for measure, *_ in MEASURES:
        aggregations[f"{measure}_mean"] = (measure, "mean")
        aggregations[f"{measure}_se"] = (measure, "sem")
        aggregations[f"{measure}_count"] = (measure, "count")

    summary = (
        frame.groupby(GROUP_COLUMNS, as_index=False, sort=True, dropna=False)
        .agg(**aggregations)
        .sort_values(GROUP_COLUMNS)
    )
    # pandas returns NaN SEM for a single observation; a zero-length error bar
    # is more useful if a partial input is deliberately plotted.
    standard_error_columns = [f"{measure}_se" for measure, *_ in MEASURES]
    summary[standard_error_columns] = summary[standard_error_columns].fillna(0.0)
    for measure, *_ in MEASURES:
        summary[f"{measure}_ci95"] = (
            NORMAL_95_PERCENT_CRITICAL_VALUE * summary[f"{measure}_se"]
        )
    return summary


def save_chart_data(summary: pd.DataFrame, output_path: Path) -> None:
    """Save the means and standard errors underlying every chart datapoint."""
    output = summary.rename(
        columns={
            "average_actuals_mean": "average_actuals",
            "average_unsmoothed_predictions_mean": "average_unsmoothed_prediction",
            "average_smoothed_predictions_mean": "average_smoothed_prediction",
            "average_actuals_se": "std_error_average_actuals",
            "average_unsmoothed_predictions_se": (
                "std_error_unsmoothed_predictions"
            ),
            "average_smoothed_predictions_se": "std_error_smoothed_predictions",
        }
    )
    output_columns = [
        "dataset",
        "depth",
        "class",
        "decile_bucket",
        "average_actuals",
        "average_unsmoothed_prediction",
        "average_smoothed_prediction",
        "std_error_average_actuals",
        "std_error_unsmoothed_predictions",
        "std_error_smoothed_predictions",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.loc[:, output_columns].to_csv(output_path, index=False)
    print(f"Saved chart datapoints to {output_path}", flush=True)


def report_seed_counts(summary: pd.DataFrame) -> None:
    combination_columns = ["dataset", "depth", "class"]
    seed_ranges = (
        summary.groupby(combination_columns, dropna=False)["seed_count"]
        .agg(["min", "max"])
        .reset_index()
    )
    incomplete = seed_ranges[
        (seed_ranges["min"] != EXPECTED_SEED_COUNT)
        | (seed_ranges["max"] != EXPECTED_SEED_COUNT)
    ]
    for row in incomplete.itertuples(index=False):
        class_text = "" if row._2 == "" else f", class={row._2}"
        print(
            f"Warning: dataset={row.dataset}, depth={row.depth}{class_text} has "
            f"{row.min}-{row.max} seeds per bucket; expected {EXPECTED_SEED_COUNT}.",
            flush=True,
        )


def plot_combination(
    chart_data: pd.DataFrame,
    dataset: str,
    depth: int,
    class_label: str,
    output_dir: Path,
    dpi: int,
) -> Path:
    chart_data = chart_data.sort_values("decile_bucket")
    fig, axis = plt.subplots(figsize=(9.5, 6))

    for measure, label, color, marker, linestyle in MEASURES:
        axis.errorbar(
            chart_data["decile_bucket"],
            chart_data[f"{measure}_mean"],
            yerr=chart_data[f"{measure}_ci95"],
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=6,
            capsize=4,
            elinewidth=1.2,
            markeredgecolor=color,
            markerfacecolor="white" if measure != "average_actuals" else color,
        )

    class_title = "" if class_label == "" else f" | Class {class_label}"
    axis.set_title(f"{display_dataset_name(dataset)} | Depth {depth}{class_title}")
    axis.set_xlabel("Decile bucket (lowest to highest unsmoothed prediction)")
    axis.set_ylabel("Mean across random seeds (95% confidence interval)")
    axis.set_xticks(range(1, 11))
    axis.grid(axis="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()

    class_suffix = "" if class_label == "" else f"_class_{safe_filename_part(class_label)}"
    output_path = output_dir / (
        f"{safe_filename_part(dataset)}_depth_{depth}{class_suffix}_decile.png"
    )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    deciles = load_deciles(args.input_path)
    summary = summarize_across_seeds(deciles)
    report_seed_counts(summary)
    save_chart_data(summary, args.summary_output_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combinations = list(
        summary[["dataset", "depth", "class"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    print(f"Creating {len(combinations)} decile charts.", flush=True)
    for chart_number, (dataset, depth, class_label) in enumerate(combinations, start=1):
        class_text = "" if class_label == "" else f", class={class_label}"
        print(
            f"[{chart_number}/{len(combinations)}] Plotting {dataset}, "
            f"depth={depth}{class_text}",
            flush=True,
        )
        mask = (
            summary["dataset"].eq(dataset)
            & summary["depth"].eq(depth)
            & summary["class"].eq(class_label)
        )
        output_path = plot_combination(
            summary.loc[mask],
            dataset,
            int(depth),
            class_label,
            args.output_dir,
            args.dpi,
        )
        print(f"  Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()

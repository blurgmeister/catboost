#!/usr/bin/env python3
"""Summarise smoothed-to-unsmoothed CatBoost evaluation metric ratios."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUT_PATH = Path(
    "/home/kerith/workspaces/smooth_multi/eval_metrics/eval_metrics_summary.csv"
)
DEFAULT_OUTPUT_PATH = Path(
    "/home/kerith/workspaces/smooth_multi/eval_metrics/eval_metric_ratio_summary.csv"
)
DEFAULT_STATS_OUTPUT_PATH = Path(
    "/home/kerith/workspaces/smooth_multi/eval_metrics/"
    "eval_metric_ratio_summary_stats.csv"
)

SMOOTHED_SERIES = "catboost_smoothed"
UNSMOOTHED_SERIES = "catboost_unsmoothed"
METRIC_COLUMNS = (
    "mean_train_eval_metric",
    "mean_validation_eval_metric",
)
OUTPUT_COLUMNS = (
    "dataset",
    "depth",
    "mean_train_eval_metric_ratio",
    "mean_validation_eval_metric_ratio",
)
RATIO_COLUMNS = OUTPUT_COLUMNS[2:]
STATS_OUTPUT_COLUMNS = (
    "depth",
    *(f"{ratio}_{statistic}" for ratio in RATIO_COLUMNS for statistic in (
        "median",
        "mean",
        "percentile_25",
        "percentile_75",
        "iqr",
    )),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate CatBoost smoothed/unsmoothed mean evaluation metric "
            "ratios by dataset and depth."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=DEFAULT_STATS_OUTPUT_PATH,
        help="Path for the by-depth ratio statistics CSV.",
    )
    return parser.parse_args()


def read_catboost_rows(input_path: Path) -> dict[tuple[str, str], dict[str, dict[str, str]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = {}

    with input_path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        required_columns = {"dataset", "depth", "series", *METRIC_COLUMNS}
        missing_columns = sorted(required_columns.difference(reader.fieldnames or ()))
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {', '.join(missing_columns)}"
            )

        for row_number, row in enumerate(reader, start=2):
            series = row["series"]
            if series not in {SMOOTHED_SERIES, UNSMOOTHED_SERIES}:
                continue

            key = (row["dataset"], row["depth"])
            series_rows = grouped.setdefault(key, {})
            if series in series_rows:
                raise ValueError(
                    f"Duplicate {series!r} row for dataset={key[0]!r}, "
                    f"depth={key[1]!r} at input row {row_number}"
                )
            series_rows[series] = row

    return grouped


def calculate_ratios(
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]],
) -> list[dict[str, str | float]]:
    summary: list[dict[str, str | float]] = []

    for (dataset, depth), series_rows in grouped.items():
        missing_series = {
            SMOOTHED_SERIES,
            UNSMOOTHED_SERIES,
        }.difference(series_rows)
        if missing_series:
            raise ValueError(
                f"Missing {', '.join(sorted(missing_series))} for "
                f"dataset={dataset!r}, depth={depth!r}"
            )

        output_row: dict[str, str | float] = {
            "dataset": dataset,
            "depth": depth,
        }
        for metric in METRIC_COLUMNS:
            try:
                smoothed = float(series_rows[SMOOTHED_SERIES][metric])
                unsmoothed = float(series_rows[UNSMOOTHED_SERIES][metric])
            except ValueError as error:
                raise ValueError(
                    f"Non-numeric {metric!r} for dataset={dataset!r}, "
                    f"depth={depth!r}"
                ) from error

            if unsmoothed == 0:
                raise ZeroDivisionError(
                    f"Cannot calculate {metric!r} ratio for dataset={dataset!r}, "
                    f"depth={depth!r}: the unsmoothed value is zero"
                )
            output_row[f"{metric}_ratio"] = smoothed / unsmoothed

        summary.append(output_row)

    def sort_key(row: dict[str, str | float]) -> tuple[str, float | str]:
        depth = str(row["depth"])
        try:
            return str(row["dataset"]), float(depth)
        except ValueError:
            return str(row["dataset"]), depth

    return sorted(summary, key=sort_key)


def percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for sorted numeric values."""
    if not values:
        raise ValueError("Cannot calculate a percentile from no values")

    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]

    fraction = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


def calculate_depth_stats(
    summary: list[dict[str, str | float]],
) -> list[dict[str, str | float]]:
    ratios_by_depth: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {ratio: [] for ratio in RATIO_COLUMNS}
    )
    for row in summary:
        depth = str(row["depth"])
        for ratio in RATIO_COLUMNS:
            ratios_by_depth[depth][ratio].append(float(row[ratio]))

    stats_summary: list[dict[str, str | float]] = []
    for depth, ratios in ratios_by_depth.items():
        output_row: dict[str, str | float] = {"depth": depth}
        for ratio, values in ratios.items():
            percentile_25 = percentile(values, 0.25)
            percentile_75 = percentile(values, 0.75)
            output_row[f"{ratio}_median"] = statistics.median(values)
            output_row[f"{ratio}_mean"] = statistics.fmean(values)
            output_row[f"{ratio}_percentile_25"] = percentile_25
            output_row[f"{ratio}_percentile_75"] = percentile_75
            output_row[f"{ratio}_iqr"] = percentile_75 - percentile_25
        stats_summary.append(output_row)

    return sorted(stats_summary, key=lambda row: float(row["depth"]))


def write_summary(
    summary: list[dict[str, str | float]],
    output_path: Path,
    fieldnames: tuple[str, ...],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    args = parse_args()
    grouped = read_catboost_rows(args.input)
    summary = calculate_ratios(grouped)
    stats_summary = calculate_depth_stats(summary)
    write_summary(summary, args.output, OUTPUT_COLUMNS)
    write_summary(stats_summary, args.stats_output, STATS_OUTPUT_COLUMNS)
    print(f"Wrote {len(summary)} dataset/depth rows to {args.output}")
    print(f"Wrote {len(stats_summary)} depth summary rows to {args.stats_output}")


if __name__ == "__main__":
    main()

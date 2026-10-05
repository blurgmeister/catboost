#!/usr/bin/env python3
"""Calculate decile prediction differences and fitted-line rotation metrics.

The mean absolute percentage difference is stored as a proportion and is
calculated as ``mean(abs(smoothed / unsmoothed - 1))`` across decile buckets.
For example, 0.01 represents a 1% mean absolute difference.

The slope angle difference is signed and measured in degrees as
``atan(smoothed slope) - atan(unsmoothed slope)``.

The endpoint angle difference applies the same calculation to slopes of lines
passing through the first- and tenth-decile coordinates only.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_INPUT_PATH = Path(
    "/home/kerith/workspaces/smooth_multi/decile/decile_summary.csv"
)
DEFAULT_OUTPUT_PATH = Path(
    "/home/kerith/workspaces/smooth_multi/decile/decile_rotation.csv"
)
DEFAULT_SUMMARY_OUTPUT_PATH = Path(
    "/home/kerith/workspaces/smooth_multi/decile/decile_rotation_summary.csv"
)
DEFAULT_ALL_SUMMARY_OUTPUT_PATH = Path(
    "/home/kerith/workspaces/smooth_multi/decile/decile_rotation_summary_all.csv"
)
DEFAULT_DEPTH_SUMMARY_OUTPUT_PATH = Path(
    "/home/kerith/workspaces/smooth_multi/decile/decile_rotation_summary_depth.csv"
)
DEFAULT_DATASET_SUMMARY_OUTPUT_PATH = Path(
    "/home/kerith/workspaces/smooth_multi/decile/"
    "decile_rotation_summary_by_dataset.csv"
)
DEFAULT_PLOT_OUTPUT_DIR = Path("/home/kerith/workspaces/smooth_multi/decile")

GROUP_COLUMNS = ("dataset", "depth", "class", "random_seed")
SUMMARY_GROUP_COLUMNS = ("dataset", "depth", "class")
REQUIRED_COLUMNS = (
    *GROUP_COLUMNS,
    "decile_bucket",
    "average_unsmoothed_predictions",
    "average_smoothed_predictions",
)
METRIC_COLUMNS = (
    "mean_absolute_percentage_difference",
    "unsmoothed_intercept",
    "unsmoothed_slope",
    "smoothed_intercept",
    "smoothed_slope",
    "slope_ratio",
    "slope_angle_difference_degrees",
    "endpoint_slope_angle_difference_degrees",
)
OUTPUT_COLUMNS = (*GROUP_COLUMNS, *METRIC_COLUMNS)
SUMMARY_METRICS = (
    "mean_absolute_percentage_difference",
    "slope_ratio",
    "slope_angle_difference_degrees",
    "endpoint_slope_angle_difference_degrees",
)
SUMMARY_OUTPUT_COLUMNS = (
    *SUMMARY_GROUP_COLUMNS,
    *(f"{metric}_{statistic}" for metric in SUMMARY_METRICS for statistic in (
        "mean",
        "median",
        "standard_error",
    )),
)
DATASET_SUMMARY_OUTPUT_COLUMNS = ("dataset", *SUMMARY_OUTPUT_COLUMNS[3:])
BROAD_SUMMARY_METRICS = (
    "mean_absolute_percentage_difference",
    "slope_ratio",
    "slope_angle_difference_degrees",
    "endpoint_slope_angle_difference_degrees",
)
BROAD_SUMMARY_STATISTICS = (
    "mean",
    "median",
    "standard_error",
    "interquartile_range",
    "percentile_2_5",
    "percentile_5",
    "percentile_25",
    "percentile_75",
    "percentile_90",
    "percentile_95",
    "percentile_97_5",
)
ALL_SUMMARY_OUTPUT_COLUMNS = tuple(
    f"{metric}_{statistic}"
    for metric in BROAD_SUMMARY_METRICS
    for statistic in BROAD_SUMMARY_STATISTICS
)
DEPTH_SUMMARY_OUTPUT_COLUMNS = ("depth", *ALL_SUMMARY_OUTPUT_COLUMNS)
PLOT_METRICS = (
    (
        "mean_absolute_percentage_difference",
        "Mean absolute prediction difference",
        "Mean absolute percentage difference (proportion)",
    ),
    ("slope_ratio", "Fitted slope ratio", "Smoothed / unsmoothed slope"),
    (
        "slope_angle_difference_degrees",
        "Fitted slope angle difference",
        "Angle difference (degrees)",
    ),
    (
        "endpoint_slope_angle_difference_degrees",
        "Decile 1-to-10 slope angle difference",
        "Endpoint angle difference (degrees)",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate prediction differences and fitted-line rotation metrics "
            "from a decile summary CSV."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT_PATH
    )
    parser.add_argument(
        "--all-summary-output", type=Path, default=DEFAULT_ALL_SUMMARY_OUTPUT_PATH
    )
    parser.add_argument(
        "--depth-summary-output",
        type=Path,
        default=DEFAULT_DEPTH_SUMMARY_OUTPUT_PATH,
    )
    parser.add_argument(
        "--dataset-summary-output",
        type=Path,
        default=DEFAULT_DATASET_SUMMARY_OUTPUT_PATH,
    )
    parser.add_argument(
        "--plot-output-dir",
        type=Path,
        default=DEFAULT_PLOT_OUTPUT_DIR,
        help="Directory to receive the four depth scatter plots.",
    )
    return parser.parse_args()


def finite_float(value: str, column: str, row_number: int) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(
            f"Input row {row_number} has a non-numeric {column!r}: {value!r}"
        ) from error
    if not math.isfinite(result):
        raise ValueError(
            f"Input row {row_number} has a non-finite {column!r}: {value!r}"
        )
    return result


def read_groups(
    input_path: Path,
) -> dict[tuple[str, str, str, str], list[tuple[float, float, float]]]:
    """Read and group x, unsmoothed y, and smoothed y values."""
    grouped: dict[
        tuple[str, str, str, str], list[tuple[float, float, float]]
    ] = defaultdict(list)

    with input_path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        missing = sorted(set(REQUIRED_COLUMNS).difference(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        for row_number, row in enumerate(reader, start=2):
            key = tuple(row[column] for column in GROUP_COLUMNS)
            grouped[key].append(
                (
                    finite_float(row["decile_bucket"], "decile_bucket", row_number),
                    finite_float(
                        row["average_unsmoothed_predictions"],
                        "average_unsmoothed_predictions",
                        row_number,
                    ),
                    finite_float(
                        row["average_smoothed_predictions"],
                        "average_smoothed_predictions",
                        row_number,
                    ),
                )
            )

    if not grouped:
        raise ValueError(f"Input CSV contains no data rows: {input_path}")
    return dict(grouped)


def fit_line(x_values: Sequence[float], y_values: Sequence[float]) -> tuple[float, float]:
    """Return the ordinary-least-squares intercept and slope for y ~ x."""
    if len(x_values) != len(y_values) or not x_values:
        raise ValueError("Line fitting requires equally sized, non-empty x and y values")

    mean_x = statistics.fmean(x_values)
    mean_y = statistics.fmean(y_values)
    denominator = math.fsum((x - mean_x) ** 2 for x in x_values)
    if denominator == 0:
        raise ValueError("Cannot fit a line when all decile_bucket values are equal")
    slope = math.fsum(
        (x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values)
    ) / denominator
    return mean_y - slope * mean_x, slope


def endpoint_slope(
    values_by_bucket: dict[float, tuple[float, float]],
    prediction_index: int,
    context: str,
) -> float:
    """Return the slope through prediction values at deciles 1 and 10."""
    missing = [bucket for bucket in (1.0, 10.0) if bucket not in values_by_bucket]
    if missing:
        raise ValueError(
            f"Missing endpoint decile buckets {missing} for {context}"
        )
    first_prediction = values_by_bucket[1.0][prediction_index]
    tenth_prediction = values_by_bucket[10.0][prediction_index]
    return (tenth_prediction - first_prediction) / (10.0 - 1.0)


def numeric_sort_key(value: str) -> tuple[int, float | str]:
    try:
        return 0, float(value)
    except ValueError:
        return 1, value


def row_sort_key(row: dict[str, str | float]) -> tuple[object, ...]:
    return (
        str(row["dataset"]),
        numeric_sort_key(str(row["depth"])),
        str(row["class"]),
        numeric_sort_key(str(row.get("random_seed", ""))),
    )


def calculate_rotation_rows(
    grouped: dict[tuple[str, str, str, str], list[tuple[float, float, float]]],
) -> list[dict[str, str | float]]:
    output: list[dict[str, str | float]] = []
    for key, values in grouped.items():
        dataset, depth, class_name, random_seed = key
        x_values = [value[0] for value in values]
        unsmoothed = [value[1] for value in values]
        smoothed = [value[2] for value in values]

        duplicate_buckets = sorted(
            bucket for bucket in set(x_values) if x_values.count(bucket) > 1
        )
        if duplicate_buckets:
            raise ValueError(
                f"Duplicate decile buckets {duplicate_buckets} for "
                f"dataset={dataset!r}, depth={depth!r}, class={class_name!r}, "
                f"random_seed={random_seed!r}"
            )
        context = (
            f"dataset={dataset!r}, depth={depth!r}, class={class_name!r}, "
            f"random_seed={random_seed!r}"
        )
        values_by_bucket = {
            bucket: (unsmoothed_value, smoothed_value)
            for bucket, unsmoothed_value, smoothed_value in values
        }
        if any(value == 0 for value in unsmoothed):
            raise ZeroDivisionError(
                "Cannot calculate percentage difference because an average "
                f"unsmoothed prediction is zero for dataset={dataset!r}, "
                f"depth={depth!r}, class={class_name!r}, random_seed={random_seed!r}"
            )

        mean_absolute_percentage_difference = statistics.fmean(
            abs(smoothed_value / unsmoothed_value - 1.0)
            for smoothed_value, unsmoothed_value in zip(smoothed, unsmoothed)
        )
        unsmoothed_intercept, unsmoothed_slope = fit_line(x_values, unsmoothed)
        smoothed_intercept, smoothed_slope = fit_line(x_values, smoothed)
        unsmoothed_endpoint_slope = endpoint_slope(values_by_bucket, 0, context)
        smoothed_endpoint_slope = endpoint_slope(values_by_bucket, 1, context)
        if unsmoothed_slope == 0:
            raise ZeroDivisionError(
                "Cannot calculate slope ratio because the unsmoothed fitted "
                f"slope is zero for dataset={dataset!r}, depth={depth!r}, "
                f"class={class_name!r}, random_seed={random_seed!r}"
            )

        output.append(
            {
                "dataset": dataset,
                "depth": depth,
                "class": class_name,
                "random_seed": random_seed,
                "mean_absolute_percentage_difference": (
                    mean_absolute_percentage_difference
                ),
                "unsmoothed_intercept": unsmoothed_intercept,
                "unsmoothed_slope": unsmoothed_slope,
                "smoothed_intercept": smoothed_intercept,
                "smoothed_slope": smoothed_slope,
                "slope_ratio": smoothed_slope / unsmoothed_slope,
                "slope_angle_difference_degrees": math.degrees(
                    math.atan(smoothed_slope) - math.atan(unsmoothed_slope)
                ),
                "endpoint_slope_angle_difference_degrees": math.degrees(
                    math.atan(smoothed_endpoint_slope)
                    - math.atan(unsmoothed_endpoint_slope)
                ),
            }
        )
    return sorted(output, key=row_sort_key)


def standard_error(values: Sequence[float]) -> float:
    """Return the sample standard error, or zero for a single observation."""
    if not values:
        raise ValueError("Cannot calculate standard error from no values")
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile using the (n - 1) method."""
    if not values:
        raise ValueError("Cannot calculate a percentile from no values")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")

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


def calculate_summary_rows(
    rotation_rows: Iterable[dict[str, str | float]],
    group_columns: Sequence[str] = SUMMARY_GROUP_COLUMNS,
) -> list[dict[str, str | float]]:
    grouped: dict[
        tuple[str, ...], dict[str, list[float]]
    ] = defaultdict(lambda: {metric: [] for metric in SUMMARY_METRICS})
    for row in rotation_rows:
        key = tuple(str(row[column]) for column in group_columns)
        for metric in SUMMARY_METRICS:
            grouped[key][metric].append(float(row[metric]))

    output: list[dict[str, str | float]] = []
    for key, metrics in grouped.items():
        row: dict[str, str | float] = dict(zip(group_columns, key))
        for metric, values in metrics.items():
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_median"] = statistics.median(values)
            row[f"{metric}_standard_error"] = standard_error(values)
        output.append(row)
    if tuple(group_columns) == SUMMARY_GROUP_COLUMNS:
        return sorted(output, key=row_sort_key)
    return sorted(
        output, key=lambda row: tuple(str(row[column]) for column in group_columns)
    )


def calculate_broad_summary_rows(
    rotation_rows: Iterable[dict[str, str | float]],
    group_columns: Sequence[str],
) -> list[dict[str, str | float]]:
    """Summarise rotation metrics either globally or by the given columns."""
    grouped: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(
        lambda: {metric: [] for metric in BROAD_SUMMARY_METRICS}
    )
    for row in rotation_rows:
        key = tuple(str(row[column]) for column in group_columns)
        for metric in BROAD_SUMMARY_METRICS:
            grouped[key][metric].append(float(row[metric]))

    output: list[dict[str, str | float]] = []
    for key, metrics in grouped.items():
        row: dict[str, str | float] = dict(zip(group_columns, key))
        for metric, values in metrics.items():
            percentile_25 = percentile(values, 0.25)
            percentile_75 = percentile(values, 0.75)
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_median"] = statistics.median(values)
            row[f"{metric}_standard_error"] = standard_error(values)
            row[f"{metric}_interquartile_range"] = percentile_75 - percentile_25
            row[f"{metric}_percentile_2_5"] = percentile(values, 0.025)
            row[f"{metric}_percentile_5"] = percentile(values, 0.05)
            row[f"{metric}_percentile_25"] = percentile_25
            row[f"{metric}_percentile_75"] = percentile_75
            row[f"{metric}_percentile_90"] = percentile(values, 0.90)
            row[f"{metric}_percentile_95"] = percentile(values, 0.95)
            row[f"{metric}_percentile_97_5"] = percentile(values, 0.975)
        output.append(row)

    if tuple(group_columns) == ("depth",):
        return sorted(output, key=lambda row: numeric_sort_key(str(row["depth"])))
    return output


def write_rows(
    output_path: Path,
    rows: Iterable[dict[str, str | float]],
    columns: Sequence[str],
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            row_count += 1
    return row_count


def read_plot_rows(
    input_path: Path, required_columns: Sequence[str]
) -> list[dict[str, str]]:
    """Read a generated summary CSV and validate its plotting columns."""
    with input_path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        missing = sorted(set(required_columns).difference(reader.fieldnames or ()))
        if missing:
            raise ValueError(
                f"Missing plotting columns in {input_path}: {', '.join(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"Plotting input contains no data rows: {input_path}")
    return rows


def create_rotation_plots(
    summary_path: Path,
    depth_summary_path: Path,
    output_dir: Path,
) -> list[Path]:
    """Plot dataset/class values with global depth-level summary series."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Creating rotation plots requires matplotlib. Run this script in "
            "the project development environment."
        ) from error

    summary_columns = ["dataset", "depth", "class"]
    depth_summary_columns = ["depth"]
    for metric, _, _ in PLOT_METRICS:
        summary_columns.extend((f"{metric}_mean", f"{metric}_median"))
        depth_summary_columns.extend((f"{metric}_mean", f"{metric}_median"))

    summary_rows = read_plot_rows(summary_path, summary_columns)
    depth_summary_rows = read_plot_rows(depth_summary_path, depth_summary_columns)
    depth_summary_rows.sort(key=lambda row: float(row["depth"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    depth_values = [float(row["depth"]) for row in summary_rows]
    unique_depths = sorted(set(depth_values))
    for metric, title, y_label in PLOT_METRICS:
        mean_column = f"{metric}_mean"
        median_column = f"{metric}_median"
        mean_values = [float(row[mean_column]) for row in summary_rows]
        median_values = [float(row[median_column]) for row in summary_rows]
        global_depths = [float(row["depth"]) for row in depth_summary_rows]
        global_means = [float(row[mean_column]) for row in depth_summary_rows]
        global_medians = [float(row[median_column]) for row in depth_summary_rows]

        numeric_values = [
            *mean_values,
            *median_values,
            *global_means,
            *global_medians,
        ]
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError(f"Non-finite plotting value found for {metric}")

        figure, axis = plt.subplots(figsize=(11, 7))
        axis.scatter(
            depth_values,
            mean_values,
            marker="o",
            s=34,
            alpha=0.45,
            color="#1f77b4",
            edgecolors="none",
            label="Dataset/class mean",
        )
        axis.scatter(
            depth_values,
            median_values,
            marker="x",
            s=38,
            alpha=0.6,
            color="#ff7f0e",
            linewidths=1.2,
            label="Dataset/class median",
        )
        axis.plot(
            global_depths,
            global_means,
            color="#174a6e",
            linestyle="--",
            linewidth=2,
            marker="D",
            markersize=5,
            label="Global mean by depth",
        )
        axis.plot(
            global_depths,
            global_medians,
            color="#a64b00",
            linestyle=":",
            linewidth=2,
            marker="s",
            markersize=5,
            label="Global median by depth",
        )
        axis.set_title(f"{title} by tree depth")
        axis.set_xlabel("Tree depth")
        axis.set_ylabel(y_label)
        axis.set_xticks(unique_depths)
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.tight_layout()

        output_path = output_dir / f"scatter_{metric}_by_depth.png"
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(output_path)

    return output_paths


def main() -> None:
    args = parse_args()
    grouped = read_groups(args.input)
    rotation_rows = calculate_rotation_rows(grouped)
    summary_rows = calculate_summary_rows(rotation_rows)
    dataset_summary_rows = calculate_summary_rows(rotation_rows, ("dataset",))
    all_summary_rows = calculate_broad_summary_rows(rotation_rows, ())
    depth_summary_rows = calculate_broad_summary_rows(rotation_rows, ("depth",))
    rotation_count = write_rows(args.output, rotation_rows, OUTPUT_COLUMNS)
    summary_count = write_rows(
        args.summary_output, summary_rows, SUMMARY_OUTPUT_COLUMNS
    )
    dataset_summary_count = write_rows(
        args.dataset_summary_output,
        dataset_summary_rows,
        DATASET_SUMMARY_OUTPUT_COLUMNS,
    )
    all_summary_count = write_rows(
        args.all_summary_output, all_summary_rows, ALL_SUMMARY_OUTPUT_COLUMNS
    )
    depth_summary_count = write_rows(
        args.depth_summary_output,
        depth_summary_rows,
        DEPTH_SUMMARY_OUTPUT_COLUMNS,
    )
    print(f"Wrote {rotation_count:,} seed-level rows to {args.output}")
    print(f"Wrote {summary_count:,} seed-summary rows to {args.summary_output}")
    print(
        f"Wrote {dataset_summary_count:,} dataset summary rows to "
        f"{args.dataset_summary_output}"
    )
    print(
        f"Wrote {all_summary_count:,} overall summary row to "
        f"{args.all_summary_output}"
    )
    print(
        f"Wrote {depth_summary_count:,} depth summary rows to "
        f"{args.depth_summary_output}"
    )
    plot_paths = create_rotation_plots(
        args.summary_output, args.depth_summary_output, args.plot_output_dir
    )
    for plot_path in plot_paths:
        print(f"Saved scatter plot to {plot_path}")


if __name__ == "__main__":
    main()

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PREDICTION_DIR = Path("/home/kerith/workspaces/smooth_multi/predictions")
METADATA_PATH = Path("/home/kerith/workspaces/smooth_multi/dataset_metadata.csv")
OUTPUT_DIR = Path("/home/kerith/workspaces/smooth_multi")
DEFAULT_PERCENTILE_OUTPUT_PATH = (
    OUTPUT_DIR / "prediction_difference_width_ratio_by_numeric_proportion.png"
)
DEFAULT_STD_OUTPUT_PATH = (
    OUTPUT_DIR / "prediction_difference_std_ratio_by_numeric_proportion.png"
)
DEFAULT_PERCENTILE_ROWS_OUTPUT_PATH = (
    OUTPUT_DIR / "prediction_difference_width_ratio_by_dataset_rows.png"
)
DEFAULT_STD_ROWS_OUTPUT_PATH = (
    OUTPUT_DIR / "prediction_difference_std_ratio_by_dataset_rows.png"
)
DEFAULT_PERCENTILE_DEPTH_OUTPUT_PATH = (
    OUTPUT_DIR / "prediction_difference_width_ratio_by_model_depth.png"
)
DEFAULT_STD_DEPTH_OUTPUT_PATH = (
    OUTPUT_DIR / "prediction_difference_std_ratio_by_model_depth.png"
)
DEFAULT_SUMMARY_PATH = OUTPUT_DIR / "prediction_difference_ratio_summary.csv"

MODEL_KINDS = ("smoothed", "unsmoothed")
FILENAME_RE = re.compile(
    r"(.+)_(smoothed|unsmoothed)_depth_(\d+)_predictions\.csv$"
)
PREDICTION_COLUMN_RE = re.compile(r"pred_seed_(\d+)(?:_class_(.+))?")
TASK_MARKERS = {
    "Regression": "o",
    "Binary": "^",
    "Multiclass": "s",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the smoothed/unsmoothed ratio of the 99.5th-to-0.5th "
            "percentile width of row-wise seed prediction differences."
        )
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=PREDICTION_DIR,
        help="Directory containing the *_predictions.csv files.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=METADATA_PATH,
        help="Dataset metadata CSV containing Dataset, Task, and proportion_numeric_columns.",
    )
    parser.add_argument(
        "--percentile-output",
        type=Path,
        default=DEFAULT_PERCENTILE_OUTPUT_PATH,
        help="Path for the percentile-width ratio chart.",
    )
    parser.add_argument(
        "--std-output",
        type=Path,
        default=DEFAULT_STD_OUTPUT_PATH,
        help="Path for the standard-deviation ratio chart.",
    )
    parser.add_argument(
        "--percentile-rows-output",
        type=Path,
        default=DEFAULT_PERCENTILE_ROWS_OUTPUT_PATH,
        help="Path for the percentile-width ratio chart using dataset rows.",
    )
    parser.add_argument(
        "--std-rows-output",
        type=Path,
        default=DEFAULT_STD_ROWS_OUTPUT_PATH,
        help="Path for the standard-deviation ratio chart using dataset rows.",
    )
    parser.add_argument(
        "--percentile-depth-output",
        type=Path,
        default=DEFAULT_PERCENTILE_DEPTH_OUTPUT_PATH,
        help="Path for the percentile-width ratio chart using model depth.",
    )
    parser.add_argument(
        "--std-depth-output",
        type=Path,
        default=DEFAULT_STD_DEPTH_OUTPUT_PATH,
        help="Path for the standard-deviation ratio chart using model depth.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path for the calculated widths and ratios CSV.",
    )
    return parser.parse_args()


def discover_prediction_files(prediction_dir):
    files = {}
    for path in prediction_dir.glob("*_predictions.csv"):
        match = FILENAME_RE.fullmatch(path.name)
        if match is None:
            continue
        dataset, model_kind, depth = match.groups()
        files[(dataset, model_kind, int(depth))] = path
    return files


def prediction_groups(columns):
    """Return prediction columns grouped by output/class and indexed by seed."""
    groups = {}
    for column in columns:
        match = PREDICTION_COLUMN_RE.fullmatch(column)
        if match is None:
            continue
        seed, class_label = match.groups()
        groups.setdefault(class_label or "", {})[int(seed)] = column
    return groups


def load_prediction_differences(path):
    """Load first-vs-second-seed differences separately for every output/class."""
    header = pd.read_csv(path, nrows=0).columns
    groups = prediction_groups(header)
    if not groups:
        raise ValueError(f"No prediction columns found in {path}")

    selected_columns = []
    group_columns = {}
    for output_name in sorted(groups, key=lambda value: (value != "", value)):
        seed_columns = groups[output_name]
        seeds = sorted(seed_columns)
        if len(seeds) < 2:
            continue
        seed_pair = tuple(seeds[:2])
        columns = (seed_columns[seed_pair[0]], seed_columns[seed_pair[1]])
        group_columns[output_name] = (columns, seed_pair)
        selected_columns.extend(columns)

    if not group_columns:
        raise ValueError(f"Fewer than two prediction seeds found in {path}")

    frame = pd.read_csv(path, usecols=selected_columns)
    differences = {}
    for output_name, (columns, seed_pair) in group_columns.items():
        first, second = columns
        values = (
            frame[second].to_numpy(dtype=float)
            - frame[first].to_numpy(dtype=float)
        )
        differences[output_name] = (values[np.isfinite(values)], seed_pair)
    return differences


def percentile_width(values):
    if len(values) == 0:
        return np.nan
    return np.percentile(values, 99.5) - np.percentile(values, 0.5)


def sample_standard_deviation(values):
    if len(values) < 2:
        return np.nan
    return np.std(values, ddof=1)


def normalized_dataset_name(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def read_metadata(path):
    # The supplied file contains Windows-1252 punctuation in otherwise ASCII data.
    metadata = pd.read_csv(path, encoding="cp1252")
    required = {"Dataset", "Task", "Rows", "proportion_numeric_columns"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {', '.join(sorted(missing))}")

    metadata = metadata.copy()
    metadata["metadata_key"] = metadata["Dataset"].map(normalized_dataset_name)
    duplicate_keys = metadata.loc[
        metadata["metadata_key"].duplicated(keep=False), "metadata_key"
    ].unique()
    if len(duplicate_keys):
        raise ValueError(
            "Dataset names are ambiguous after normalization: "
            + ", ".join(sorted(duplicate_keys))
        )
    return metadata.set_index("metadata_key")


def build_summary(files, metadata):
    rows = []
    datasets_and_depths = sorted(
        {(dataset, depth) for dataset, _, depth in files},
        key=lambda value: (value[0].lower(), value[1]),
    )
    for dataset, depth in datasets_and_depths:
        paths = {
            model_kind: files.get((dataset, model_kind, depth))
            for model_kind in MODEL_KINDS
        }
        if any(path is None for path in paths.values()):
            print(f"Skipping {dataset} depth {depth}: smoothed/unsmoothed pair is incomplete")
            continue

        metadata_key = normalized_dataset_name(dataset)
        if metadata_key not in metadata.index:
            print(f"Skipping {dataset} depth {depth}: dataset is absent from metadata")
            continue
        metadata_row = metadata.loc[metadata_key]
        task = str(metadata_row["Task"]).strip()
        if task not in TASK_MARKERS:
            print(f"Skipping {dataset} depth {depth}: unrecognised task {task!r}")
            continue

        differences_by_model = {}
        for model_kind, path in paths.items():
            differences_by_model[model_kind] = load_prediction_differences(path)

        smoothed_outputs = set(differences_by_model["smoothed"])
        unsmoothed_outputs = set(differences_by_model["unsmoothed"])
        if smoothed_outputs != unsmoothed_outputs:
            raise ValueError(
                f"Output/class mismatch for {dataset} depth {depth}: "
                f"{sorted(smoothed_outputs)} vs {sorted(unsmoothed_outputs)}"
            )

        for output_name in sorted(smoothed_outputs, key=lambda value: (value != "", value)):
            smoothed_differences, smoothed_seed_pair = differences_by_model[
                "smoothed"
            ][output_name]
            unsmoothed_differences, unsmoothed_seed_pair = differences_by_model[
                "unsmoothed"
            ][output_name]
            if smoothed_seed_pair != unsmoothed_seed_pair:
                raise ValueError(
                    f"Seed pair mismatch for {dataset} depth {depth}, "
                    f"class {output_name!r}: {smoothed_seed_pair} vs "
                    f"{unsmoothed_seed_pair}"
                )

            smoothed_width = percentile_width(smoothed_differences)
            unsmoothed_width = percentile_width(unsmoothed_differences)
            smoothed_std = sample_standard_deviation(smoothed_differences)
            unsmoothed_std = sample_standard_deviation(unsmoothed_differences)

            width_ratio = calculate_ratio(
                smoothed_width,
                unsmoothed_width,
                dataset,
                depth,
                output_name,
                "percentile width",
            )
            std_ratio = calculate_ratio(
                smoothed_std,
                unsmoothed_std,
                dataset,
                depth,
                output_name,
                "standard deviation",
            )

            rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "output_class": output_name,
                    "depth": depth,
                    "proportion_numeric_columns": float(
                        metadata_row["proportion_numeric_columns"]
                    ),
                    "dataset_rows": int(str(metadata_row["Rows"]).replace(",", "")),
                    "seed_pair": (
                        f"{smoothed_seed_pair[1]}_minus_{smoothed_seed_pair[0]}"
                    ),
                    "smoothed_difference_count": len(smoothed_differences),
                    "unsmoothed_difference_count": len(unsmoothed_differences),
                    "smoothed_p99_5_minus_p0_5": smoothed_width,
                    "unsmoothed_p99_5_minus_p0_5": unsmoothed_width,
                    "width_ratio_smoothed_to_unsmoothed": width_ratio,
                    "smoothed_difference_std": smoothed_std,
                    "unsmoothed_difference_std": unsmoothed_std,
                    "std_ratio_smoothed_to_unsmoothed": std_ratio,
                }
            )
    return pd.DataFrame(rows)


def calculate_ratio(numerator, denominator, dataset, depth, output_name, statistic):
    if denominator == 0.0 or not np.isfinite(denominator):
        output_label = "" if output_name == "" else f", class {output_name}"
        print(
            f"Omitting {statistic} ratio for {dataset} depth {depth}"
            f"{output_label}: unsmoothed value is zero/non-finite"
        )
        return np.nan
    return numerator / denominator


def plot_ratios(
    summary,
    ratio_column,
    title,
    xlabel,
    ylabel,
    x_column,
    output_path,
    color_by_depth=True,
):
    plot_data = summary[np.isfinite(summary[ratio_column])]
    if plot_data.empty:
        raise ValueError(f"No finite values from {ratio_column!r} are available to plot")

    depths = sorted(plot_data["depth"].unique())
    colors = plt.get_cmap("viridis", len(depths))
    depth_colors = {depth: colors(index) for index, depth in enumerate(depths)}

    fig, axis = plt.subplots(figsize=(12, 8))
    for row in plot_data.itertuples(index=False):
        axis.scatter(
            getattr(row, x_column),
            getattr(row, ratio_column),
            color=depth_colors[row.depth] if color_by_depth else "#1f77b4",
            marker=TASK_MARKERS[row.task],
            s=70,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.4,
        )

    averages = (
        plot_data.groupby(x_column, as_index=False)[ratio_column]
        .mean()
        .sort_values(x_column)
    )
    axis.scatter(
        averages[x_column],
        averages[ratio_column],
        color="red",
        marker="o",
        s=100,
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
    )

    axis.axhline(1.0, color="#666666", linestyle="--", linewidth=1.2)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    if x_column == "depth":
        axis.set_xticks(depths)

    depth_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=depth_colors[depth],
            markeredgecolor="black",
            markersize=8,
            label=str(depth),
        )
        for depth in depths
    ]
    task_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            color="#555555",
            markersize=8,
            label=task,
        )
        for task, marker in TASK_MARKERS.items()
        if task in set(plot_data["task"])
    ]
    if color_by_depth:
        depth_legend = axis.legend(
            handles=depth_handles,
            title="Model depth",
            loc="upper left",
        )
        axis.add_artist(depth_legend)
    task_legend = axis.legend(handles=task_handles, title="Task", loc="upper right")
    axis.add_artist(task_legend)
    average_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor="red",
        markeredgecolor="black",
        markersize=9,
        label="Average at each x value",
    )
    axis.legend(handles=[average_handle], loc="lower left")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    files = discover_prediction_files(args.prediction_dir)
    if not files:
        raise ValueError(f"No *_predictions.csv files found in {args.prediction_dir}")

    metadata = read_metadata(args.metadata)
    summary = build_summary(files, metadata)
    if summary.empty:
        raise ValueError("No complete prediction pairs matched the dataset metadata")

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    print(f"Saved {args.summary_output}")

    plot_ratios(
        summary,
        "width_ratio_smoothed_to_unsmoothed",
        "Smoothed / unsmoothed width of row-wise seed prediction differences",
        "Proportion of numeric columns",
        "(99.5th - 0.5th percentile width) ratio",
        "proportion_numeric_columns",
        args.percentile_output,
    )
    print(f"Saved {args.percentile_output}")

    plot_ratios(
        summary,
        "std_ratio_smoothed_to_unsmoothed",
        "Smoothed / unsmoothed standard deviation of row-wise seed prediction differences",
        "Proportion of numeric columns",
        "Standard deviation ratio",
        "proportion_numeric_columns",
        args.std_output,
    )
    print(f"Saved {args.std_output}")

    plot_ratios(
        summary,
        "width_ratio_smoothed_to_unsmoothed",
        "Smoothed / unsmoothed width of row-wise seed prediction differences",
        "Number of rows in dataset",
        "(99.5th - 0.5th percentile width) ratio",
        "dataset_rows",
        args.percentile_rows_output,
    )
    print(f"Saved {args.percentile_rows_output}")

    plot_ratios(
        summary,
        "std_ratio_smoothed_to_unsmoothed",
        "Smoothed / unsmoothed standard deviation of row-wise seed prediction differences",
        "Number of rows in dataset",
        "Standard deviation ratio",
        "dataset_rows",
        args.std_rows_output,
    )
    print(f"Saved {args.std_rows_output}")

    plot_ratios(
        summary,
        "width_ratio_smoothed_to_unsmoothed",
        "Smoothed / unsmoothed width of row-wise seed prediction differences",
        "Model depth",
        "(99.5th - 0.5th percentile width) ratio",
        "depth",
        args.percentile_depth_output,
        color_by_depth=False,
    )
    print(f"Saved {args.percentile_depth_output}")

    plot_ratios(
        summary,
        "std_ratio_smoothed_to_unsmoothed",
        "Smoothed / unsmoothed standard deviation of row-wise seed prediction differences",
        "Model depth",
        "Standard deviation ratio",
        "depth",
        args.std_depth_output,
        color_by_depth=False,
    )
    print(f"Saved {args.std_depth_output}")


if __name__ == "__main__":
    main()

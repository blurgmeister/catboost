import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PREDICTION_DIR = Path("/home/kerith/workspaces/smooth_multi/predictions")
MODEL_KINDS = ["unsmoothed", "smoothed"]
FILENAME_RE = re.compile(r"(.+)_(smoothed|unsmoothed)_depth_(\d+)_predictions\.csv$")

METRICS = [
    ("std", "Standard deviation"),
    ("iqr_75_25", "Interquartile range"),
    ("p95_5_range", "95th - 5th percentile"),
    ("p97_5_2_5_range", "97.5th - 2.5th percentile"),
    ("p99_5_0_5_range", "99.5th - 0.5th percentile"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot row-wise seed prediction difference widths for smoothed and "
            "unsmoothed CatBoost predictions by dataset and tree depth."
        )
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=PREDICTION_DIR,
        help="Directory containing *_predictions.csv files and receiving outputs.",
    )
    parser.add_argument(
        "--only-dataset",
        default=None,
        help="Only plot this exact dataset name as it appears in prediction filenames.",
    )
    parser.add_argument(
        "--only-dataset-name",
        default=None,
        help="Only plot datasets whose name contains this case-insensitive text.",
    )
    return parser.parse_args()


def discover_prediction_files(prediction_dir):
    files = {}
    for path in prediction_dir.glob("*_predictions.csv"):
        match = FILENAME_RE.fullmatch(path.name)
        if not match:
            continue
        dataset, model_kind, depth = match.groups()
        files[(dataset, model_kind, int(depth))] = path
    return files


def filter_datasets(datasets, args):
    if args.only_dataset:
        datasets = [dataset for dataset in datasets if dataset == args.only_dataset]
    if args.only_dataset_name:
        needle = args.only_dataset_name.lower()
        datasets = [dataset for dataset in datasets if needle in dataset.lower()]
    return datasets


def prediction_groups(columns):
    class_columns = {}
    for column in columns:
        match = re.fullmatch(r"pred_seed_(\d+)_class_(.+)", column)
        if match:
            seed, class_label = match.groups()
            class_columns.setdefault(class_label, {})[seed] = column

    if class_columns:
        return {
            class_label: {
                seed: seed_columns[seed]
                for seed in sorted(seed_columns, key=int)
            }
            for class_label, seed_columns in class_columns.items()
        }

    seed_columns = {}
    for column in columns:
        match = re.fullmatch(r"pred_seed_(\d+)", column)
        if match:
            seed_columns[match.group(1)] = column

    if seed_columns:
        return {
            "": {
                seed: seed_columns[seed]
                for seed in sorted(seed_columns, key=int)
            }
        }
    return {}


def available_outputs_for_dataset(files, dataset):
    outputs = set()
    for (file_dataset, _, _), path in files.items():
        if file_dataset != dataset:
            continue
        header = pd.read_csv(path, nrows=0).columns
        outputs.update(prediction_groups(header))
    return sorted(outputs, key=lambda value: (value != "", value))


def calculate_widths(values):
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        return {
            "count": 0,
            "std": np.nan,
            "iqr_75_25": np.nan,
            "p95_5_range": np.nan,
            "p97_5_2_5_range": np.nan,
            "p99_5_0_5_range": np.nan,
        }

    return {
        "count": len(finite_values),
        "std": np.std(finite_values, ddof=1) if len(finite_values) > 1 else 0.0,
        "iqr_75_25": np.percentile(finite_values, 75) - np.percentile(finite_values, 25),
        "p95_5_range": np.percentile(finite_values, 95) - np.percentile(finite_values, 5),
        "p97_5_2_5_range": np.percentile(finite_values, 97.5) - np.percentile(finite_values, 2.5),
        "p99_5_0_5_range": np.percentile(finite_values, 99.5) - np.percentile(finite_values, 0.5),
    }


def add_file_rows(rows, path, dataset, model_kind, depth, output_name):
    frame = pd.read_csv(path)
    groups = prediction_groups(frame.columns)
    if output_name not in groups:
        return

    seed_columns = groups[output_name]
    seeds = list(seed_columns)
    if len(seeds) < 2:
        return

    first_seed, second_seed = seeds[:2]
    first_column = seed_columns[first_seed]
    second_column = seed_columns[second_seed]
    differences = (
        frame[first_column].to_numpy(dtype=float)
        - frame[second_column].to_numpy(dtype=float)
    )
    widths = calculate_widths(differences)
    rows.append(
        {
            "dataset": dataset,
            "output": output_name,
            "model_kind": model_kind,
            "depth": depth,
            "seed_pair": f"{first_seed}_minus_{second_seed}",
            "first_seed": first_seed,
            "second_seed": second_seed,
            "first_prediction_column": first_column,
            "second_prediction_column": second_column,
            **widths,
        }
    )


def build_summary(files, datasets):
    rows = []
    for dataset in datasets:
        for output_name in available_outputs_for_dataset(files, dataset):
            for depth in sorted({depth for file_dataset, _, depth in files if file_dataset == dataset}):
                for model_kind in MODEL_KINDS:
                    path = files.get((dataset, model_kind, depth))
                    if path is None:
                        continue
                    add_file_rows(rows, path, dataset, model_kind, depth, output_name)
    return pd.DataFrame(rows)


def aggregate_for_charts(summary):
    metric_columns = [metric for metric, _ in METRICS]
    return (
        summary.groupby(["dataset", "output", "model_kind", "depth"], as_index=False)
        .agg(
            count=("count", "sum"),
            seed_pair_count=("seed_pair", "nunique"),
            **{metric: (metric, "mean") for metric in metric_columns},
        )
        .sort_values(["dataset", "output", "depth", "model_kind"])
    )


def add_relative_comparison(chart_data):
    metric_columns = [metric for metric, _ in METRICS]
    rows = []
    keys = ["dataset", "output", "depth"]
    for key_values, group in chart_data.groupby(keys):
        values = {row.model_kind: row for row in group.itertuples(index=False)}
        if "smoothed" not in values or "unsmoothed" not in values:
            continue
        row = dict(zip(keys, key_values))
        for metric in metric_columns:
            smoothed = getattr(values["smoothed"], metric)
            unsmoothed = getattr(values["unsmoothed"], metric)
            row[f"{metric}_smoothed"] = smoothed
            row[f"{metric}_unsmoothed"] = unsmoothed
            if np.isfinite(smoothed) and np.isfinite(unsmoothed) and unsmoothed != 0.0:
                row[f"{metric}_ratio_smoothed_to_unsmoothed"] = smoothed / unsmoothed
                row[f"{metric}_reduction_pct"] = (1.0 - smoothed / unsmoothed) * 100.0
            else:
                row[f"{metric}_ratio_smoothed_to_unsmoothed"] = np.nan
                row[f"{metric}_reduction_pct"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def safe_output_part(value):
    if value == "":
        return ""
    return "_class_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def plot_metric(dataset, output_name, chart_data, metric, label, output_dir):
    subset = chart_data[
        (chart_data["dataset"] == dataset)
        & (chart_data["output"] == output_name)
    ]
    if subset.empty:
        return

    fig, axis = plt.subplots(figsize=(9, 6))
    for model_kind in MODEL_KINDS:
        model_data = subset[subset["model_kind"] == model_kind].sort_values("depth")
        if model_data.empty:
            continue
        axis.plot(
            model_data["depth"],
            model_data[metric],
            marker="o",
            linewidth=2,
            label=model_kind,
        )

    output_label = "" if output_name == "" else f" | class {output_name}"
    axis.set_title(f"{dataset}{output_label}: {label} of seed prediction differences")
    axis.set_xlabel("Model depth")
    axis.set_ylabel(label)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()

    output_path = (
        output_dir
        / f"{dataset}{safe_output_part(output_name)}_prediction_difference_{metric}_by_depth.png"
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_ratio_metric(dataset, output_name, comparison, metric, label, output_dir):
    subset = comparison[
        (comparison["dataset"] == dataset)
        & (comparison["output"] == output_name)
    ].sort_values("depth")
    if subset.empty:
        return

    ratio_column = f"{metric}_ratio_smoothed_to_unsmoothed"
    if ratio_column not in subset:
        return

    fig, axis = plt.subplots(figsize=(9, 6))
    axis.plot(
        subset["depth"],
        subset[ratio_column],
        marker="o",
        linewidth=2,
        color="#1f77b4",
    )
    axis.axhline(1.0, color="#666666", linestyle="--", linewidth=1)

    output_label = "" if output_name == "" else f" | class {output_name}"
    axis.set_title(f"{dataset}{output_label}: smoothed / unsmoothed {label}")
    axis.set_xlabel("Model depth")
    axis.set_ylabel("Smoothed / unsmoothed")
    axis.grid(alpha=0.25)
    fig.tight_layout()

    output_path = (
        output_dir
        / f"{dataset}{safe_output_part(output_name)}_prediction_difference_ratio_{metric}_by_depth.png"
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_combined_ratio_metrics(dataset, output_name, comparison, output_dir):
    subset = comparison[
        (comparison["dataset"] == dataset)
        & (comparison["output"] == output_name)
    ].sort_values("depth")
    if subset.empty:
        return

    fig, axis = plt.subplots(figsize=(10, 6))
    for metric, label in METRICS:
        ratio_column = f"{metric}_ratio_smoothed_to_unsmoothed"
        if ratio_column not in subset:
            continue
        axis.plot(
            subset["depth"],
            subset[ratio_column],
            marker="o",
            linewidth=2,
            label=label,
        )
    axis.axhline(1.0, color="#666666", linestyle="--", linewidth=1)

    output_label = "" if output_name == "" else f" | class {output_name}"
    axis.set_title(f"{dataset}{output_label}: smoothed / unsmoothed ratios")
    axis.set_xlabel("Model depth")
    axis.set_ylabel("Smoothed / unsmoothed")
    axis.grid(alpha=0.25)
    axis.legend(title="Metric", loc="best")
    fig.tight_layout()

    output_path = (
        output_dir
        / f"{dataset}{safe_output_part(output_name)}_prediction_difference_ratio_all_metrics_by_depth.png"
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    args = parse_args()
    files = discover_prediction_files(args.prediction_dir)
    datasets = sorted({dataset for dataset, _, _ in files})
    datasets = filter_datasets(datasets, args)
    if not datasets:
        print("No datasets matched the requested filters.")
        return

    summary = build_summary(files, datasets)
    if summary.empty:
        print("No prediction columns were found in the matched files.")
        return

    summary_path = args.prediction_dir / "prediction_difference_widths.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    chart_data = aggregate_for_charts(summary)
    chart_data_path = args.prediction_dir / "prediction_difference_widths_chart_data.csv"
    chart_data.to_csv(chart_data_path, index=False)
    print(f"Saved {chart_data_path}")

    comparison = add_relative_comparison(chart_data)
    comparison_path = args.prediction_dir / "prediction_difference_width_reductions.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"Saved {comparison_path}")

    for dataset in sorted(chart_data["dataset"].unique()):
        dataset_outputs = chart_data.loc[chart_data["dataset"] == dataset, "output"].unique()
        for output_name in sorted(dataset_outputs, key=lambda value: (value != "", value)):
            for metric, label in METRICS:
                plot_metric(dataset, output_name, chart_data, metric, label, args.prediction_dir)
                plot_ratio_metric(
                    dataset,
                    output_name,
                    comparison,
                    metric,
                    label,
                    args.prediction_dir,
                )
            plot_combined_ratio_metrics(dataset, output_name, comparison, args.prediction_dir)


if __name__ == "__main__":
    main()

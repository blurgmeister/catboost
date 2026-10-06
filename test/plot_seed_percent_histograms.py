"""Summarize and plot seed percentage-change histograms.

The input files are produced by ``calc_seed_percentage_histograms.py``. Each
seed-pair column contains weighted counts for one percentage-change histogram.
This script summarizes those columns at every bucket, then plots the mean
histogram and its 95% confidence band for smoothed and unsmoothed models.
"""

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


PREDICTION_DIR = Path("/home/kerith/workspaces/smooth_multi/predictions")
DEPTHS = [1, 2, 3, 4, 5, 6]
MODEL_KINDS = ["unsmoothed", "smoothed"]
MODEL_COLORS = {"unsmoothed": "#1f77b4", "smoothed": "#ff7f0e"}
INPUT_RE = re.compile(
    r"(.+)_(smoothed|unsmoothed)_depth_(\d+)_percent_histogram\.csv$"
)
SEED_PAIR_RE = re.compile(r"^seed_\d+_v_seed_\d+$")
CONFIDENCE_Z = 1.96


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize seed percentage-change histograms and plot smoothed "
            "versus unsmoothed mean histograms by depth."
        )
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=PREDICTION_DIR,
        help="Directory containing percent-histogram CSVs and receiving outputs.",
    )
    parser.add_argument(
        "--only-dataset",
        default=None,
        help="Only process this exact dataset name.",
    )
    parser.add_argument(
        "--only-dataset-name",
        default=None,
        help="Only process datasets containing this case-insensitive text.",
    )
    return parser.parse_args()


def discover_histogram_files(prediction_dir):
    files = {}
    for path in prediction_dir.glob("*_percent_histogram.csv"):
        match = INPUT_RE.fullmatch(path.name)
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


def summarize_histogram(frame, path):
    missing = {"bin_lower", "bin_upper"}.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")

    seed_pair_columns = [
        column for column in frame.columns if SEED_PAIR_RE.fullmatch(column)
    ]
    if not seed_pair_columns:
        raise ValueError(f"No seed-pair histogram columns found in {path}")

    values = frame[seed_pair_columns].to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    finite_counts = np.isfinite(values).sum(axis=1)
    if np.any(finite_counts == 0):
        raise ValueError(f"At least one histogram row has no finite counts in {path}")

    means = np.nanmean(values, axis=1)
    sample_std = np.nanstd(values, axis=1, ddof=1)
    standard_error = sample_std / np.sqrt(finite_counts)
    standard_error[finite_counts == 1] = 0.0
    percentiles = np.nanpercentile(values, [5, 25, 50, 75, 95], axis=1).T

    summary = frame[["bin_lower", "bin_upper"]].copy()
    if "class" in frame.columns:
        summary.insert(0, "class", frame["class"])
    summary["seed_pair_count"] = finite_counts
    summary["mean"] = means
    summary["std_error"] = standard_error
    summary["percentile_05"] = percentiles[:, 0]
    summary["percentile_25"] = percentiles[:, 1]
    summary["percentile_50"] = percentiles[:, 2]
    summary["percentile_75"] = percentiles[:, 3]
    summary["percentile_95"] = percentiles[:, 4]
    summary["interquartile_range"] = (
        summary["percentile_75"] - summary["percentile_25"]
    )
    return summary


def summary_path_for(histogram_path):
    return histogram_path.with_name(
        histogram_path.name.replace(
            "_percent_histogram.csv", "_percent_hist_summary.csv"
        )
    )


def build_summaries(files):
    summaries = {}
    for key, path in sorted(files.items()):
        summary = summarize_histogram(pd.read_csv(path), path)
        output_path = summary_path_for(path)
        summary.to_csv(output_path, index=False)
        print(f"Saved {output_path}")
        summaries[key] = summary
    return summaries


def outputs_for_dataset(summaries, dataset):
    class_names = set()
    has_multiclass_data = False
    for (file_dataset, _, _), summary in summaries.items():
        if file_dataset != dataset or "class" not in summary.columns:
            continue
        has_multiclass_data = True
        class_names.update(summary["class"].dropna().astype(str))
    return sorted(class_names) if has_multiclass_data else [None]


def summary_for_output(summary, output_name):
    if output_name is None:
        return summary
    if "class" not in summary.columns:
        return summary.iloc[0:0]
    return summary[summary["class"].astype(str) == output_name]


def safe_output_part(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def plot_dataset_output(dataset, output_name, summaries, output_dir):
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(18, 9),
        sharex=True,
        sharey=True,
    )
    axes = axes.ravel()

    class_label = "" if output_name is None else f" | class {output_name}"
    fig.suptitle(
        f"{dataset}{class_label}: weighted seed prediction percentage changes"
    )

    legend_handles = []
    legend_labels = []
    for axis, depth in zip(axes, DEPTHS):
        plotted = False
        for model_kind in MODEL_KINDS:
            summary = summaries.get((dataset, model_kind, depth))
            if summary is None:
                continue
            plot_data = summary_for_output(summary, output_name).sort_values(
                "bin_lower"
            )
            if plot_data.empty:
                continue

            x = (
                plot_data["bin_lower"].to_numpy(dtype=float)
                + plot_data["bin_upper"].to_numpy(dtype=float)
            ) / 2.0
            mean = plot_data["mean"].to_numpy(dtype=float)
            total_mean = np.sum(mean)
            if not np.isfinite(total_mean) or total_mean <= 0.0:
                continue
            mean = mean / total_mean * 100.0
            margin = (
                CONFIDENCE_Z
                * plot_data["std_error"].to_numpy(dtype=float)
                / total_mean
                * 100.0
            )
            color = MODEL_COLORS[model_kind]
            line = axis.plot(
                x,
                mean,
                color=color,
                linewidth=2,
                label=model_kind,
            )[0]
            axis.fill_between(
                x,
                mean - margin,
                mean + margin,
                color=color,
                alpha=0.2,
                linewidth=0,
            )
            if model_kind not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(model_kind)
            plotted = True

        if not plotted:
            axis.set_visible(False)
            continue

        axis.set_title(f"Depth {depth}")
        axis.set_xlabel("Prediction percentage change")
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Volume of Data (%)")
    axes[3].set_ylabel("Volume of Data (%)")
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.96),
        )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    class_suffix = (
        "" if output_name is None else f"_class_{safe_output_part(output_name)}"
    )
    output_path = (
        output_dir / f"{dataset}{class_suffix}_seed_prediction_diff_hist.png"
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    args = parse_args()
    files = discover_histogram_files(args.prediction_dir)
    datasets = sorted({dataset for dataset, _, _ in files})
    datasets = filter_datasets(datasets, args)
    selected_datasets = set(datasets)
    files = {
        key: path for key, path in files.items() if key[0] in selected_datasets
    }
    if not files:
        print("No percent-histogram CSV files matched the requested inputs.")
        return

    summaries = build_summaries(files)
    for dataset in datasets:
        for output_name in outputs_for_dataset(summaries, dataset):
            plot_dataset_output(dataset, output_name, summaries, args.prediction_dir)


if __name__ == "__main__":
    main()

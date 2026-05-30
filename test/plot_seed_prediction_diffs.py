import argparse
import re
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PREDICTION_DIR = Path("/home/kerith/workspaces/smooth_multi/predictions")
DEPTHS = [1, 2, 3, 4, 5, 6]
MODEL_KINDS = ["unsmoothed", "smoothed"]
FILENAME_RE = re.compile(r"(.+)_(smoothed|unsmoothed)_depth_(\d+)_predictions\.csv$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot row-wise seed prediction difference histograms."
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


def discover_prediction_files():
    files = {}
    for path in PREDICTION_DIR.glob("*_predictions.csv"):
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
    columns = list(columns)
    class_columns = {}
    for column in columns:
        match = re.fullmatch(r"pred_seed_(\d+)_class_(.+)", column)
        if match:
            seed, class_label = match.groups()
            class_columns.setdefault(class_label, {})[seed] = column

    if class_columns:
        return {
            class_label: [seed_columns[seed] for seed in sorted(seed_columns, key=int)]
            for class_label, seed_columns in class_columns.items()
            if len(seed_columns) >= 2
        }

    seed_columns = [
        column
        for column in columns
        if re.fullmatch(r"pred_seed_\d+", column)
    ]
    seed_columns = sorted(seed_columns, key=lambda column: int(column.split("_")[-1]))
    if len(seed_columns) < 2:
        return {}
    return {"": seed_columns}


def available_outputs_for_dataset(files, dataset):
    sample_paths = [
        path
        for (file_dataset, _, _), path in files.items()
        if file_dataset == dataset
    ]
    groups = {}
    for path in sample_paths:
        header = pd.read_csv(path, nrows=0).columns
        for output_name in prediction_groups(header):
            groups[output_name] = True
    return sorted(groups, key=lambda value: (value != "", value))


def load_seed_differences(path, output_name):
    frame = pd.read_csv(path)
    groups = prediction_groups(frame.columns)
    if output_name not in groups:
        raise ValueError(f"Output {output_name!r} not found in {path}")
    first_column, second_column = groups[output_name][:2]
    diffs = frame[second_column].to_numpy(dtype=float) - frame[first_column].to_numpy(dtype=float)
    return diffs[np.isfinite(diffs)]


def add_summary(summary_rows, dataset, output_name, model_kind, depth, diffs):
    summary_rows.append(
        {
            "dataset": dataset,
            "output": output_name,
            "model_kind": model_kind,
            "depth": depth,
            "count": len(diffs),
            "mean_diff": np.mean(diffs),
            "median_diff": np.median(diffs),
            "p10_diff": np.percentile(diffs, 10),
            "p90_diff": np.percentile(diffs, 90),
            "p95_abs_diff": np.percentile(np.abs(diffs), 95),
            "p99_abs_diff": np.percentile(np.abs(diffs), 99),
            "max_abs_diff": np.max(np.abs(diffs)),
        }
    )


def plot_dataset_output(dataset, output_name, files, summary_rows):
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(18, 9),
        sharey=True,
    )
    axes = axes.ravel()

    output_label = "" if output_name == "" else f" | class {output_name}"
    fig.suptitle(f"{dataset}{output_label}: row-wise seed prediction differences")

    for axis, depth in zip(axes, DEPTHS):
        depth_diffs = {}
        for model_kind in MODEL_KINDS:
            path = files.get((dataset, model_kind, depth))
            if path is None:
                continue
            diffs = load_seed_differences(path, output_name)
            depth_diffs[model_kind] = diffs
            add_summary(summary_rows, dataset, output_name, model_kind, depth, diffs)

        if not depth_diffs:
            axis.set_visible(False)
            continue

        combined = np.concatenate(list(depth_diffs.values()))
        if len(combined) == 0:
            axis.set_visible(False)
            continue

        lower = np.percentile(combined, 1)
        upper = np.percentile(combined, 99)
        if lower == upper:
            spread = max(abs(lower), 1.0)
            lower -= spread
            upper += spread
        bins = np.linspace(lower, upper, 50)

        for model_kind in MODEL_KINDS:
            if model_kind not in depth_diffs:
                continue
            axis.hist(
                np.clip(depth_diffs[model_kind], lower, upper),
                bins=bins,
                alpha=0.55,
                density=True,
                label=model_kind,
            )

        axis.set_title(f"Depth {depth}")
        axis.set_xlabel("Prediction difference between seeds")
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Density")
    axes[3].set_ylabel("Density")
    axes[-1].legend()
    fig.tight_layout()

    class_suffix = "" if output_name == "" else f"_class_{output_name}"
    output_path = PREDICTION_DIR / f"{dataset}{class_suffix}_seed_prediction_diff_hist.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    args = parse_args()
    files = discover_prediction_files()
    datasets = sorted({dataset for dataset, _, _ in files})
    datasets = filter_datasets(datasets, args)
    if not datasets:
        print("No datasets matched the requested filters.")
        return
    summary_rows = []

    for dataset in datasets:
        for output_name in available_outputs_for_dataset(files, dataset):
            plot_dataset_output(dataset, output_name, files, summary_rows)

    summary_path = PREDICTION_DIR / "seed_prediction_diff_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()

"""Tabulate weighted seed-to-seed prediction percentage-change histograms.

For each ``*_predictions.csv`` file, prediction columns are paired without
reuse in numeric seed order: seed 1 with seed 0, seed 3 with seed 2, and so on.
Multiclass predictions are paired separately for each class.  The percentage
change for a pair is ``numerator / denominator - 1``.

The output has 100 bins of width 0.01 covering [-0.5, 0.5].  Values below or
above that range are folded into the first or last bin, respectively.  Counts
are weighted by the input file's ``w`` column.  Multiclass outputs contain one
100-row block per class, identified by the ``class`` column.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


PREDICTION_DIR = Path("/home/kerith/workspaces/smooth_multi/predictions")
INPUT_SUFFIX = "_predictions.csv"
OUTPUT_SUFFIX = "_percent_histogram.csv"
SCALAR_PREDICTION_RE = re.compile(r"^pred_seed_(\d+)$")
MULTICLASS_PREDICTION_RE = re.compile(r"^pred_seed_(\d+)_class_(.+)$")
BIN_MIN = -0.5
BIN_MAX = 0.5
BIN_WIDTH = 0.01
BIN_EDGES = np.linspace(BIN_MIN, BIN_MAX, 101)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create weighted histograms of percentage changes between "
            "non-overlapping adjacent seed prediction pairs."
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
        help=(
            "Only process files whose dataset portion exactly matches this value. "
            "The dataset portion precedes _smoothed_depth_ or _unsmoothed_depth_."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Number of input rows to process at once (default: 10000).",
    )
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be greater than zero")
    return args


def dataset_from_path(path):
    match = re.fullmatch(
        r"(.+)_(?:smoothed|unsmoothed)_depth_\d+_predictions\.csv", path.name
    )
    return match.group(1) if match else None


def discover_input_files(prediction_dir, only_dataset=None):
    paths = sorted(prediction_dir.glob(f"*{INPUT_SUFFIX}"))
    if only_dataset is not None:
        paths = [path for path in paths if dataset_from_path(path) == only_dataset]
    return paths


def prediction_groups(columns):
    """Return output/class name -> {seed number: prediction column}."""
    scalar_columns = {}
    multiclass_columns = {}

    for column in columns:
        scalar_match = SCALAR_PREDICTION_RE.fullmatch(column)
        if scalar_match:
            scalar_columns[int(scalar_match.group(1))] = column
            continue

        multiclass_match = MULTICLASS_PREDICTION_RE.fullmatch(column)
        if multiclass_match:
            seed = int(multiclass_match.group(1))
            class_label = multiclass_match.group(2)
            multiclass_columns.setdefault(class_label, {})[seed] = column

    groups = {}
    if scalar_columns:
        groups[""] = scalar_columns
    groups.update(multiclass_columns)
    return groups


def adjacent_seed_pairs(columns):
    """Pair adjacent ordered seeds within each scalar output or class."""
    pairs_by_output = {}
    for output_name, seed_columns in prediction_groups(columns).items():
        pairs = []
        ordered_seeds = sorted(seed_columns)
        for denominator_seed, numerator_seed in zip(
            ordered_seeds[::2], ordered_seeds[1::2]
        ):
            denominator = seed_columns[denominator_seed]
            numerator = seed_columns[numerator_seed]
            result_name = f"seed_{numerator_seed}_v_seed_{denominator_seed}"
            pairs.append((result_name, numerator, denominator))
        if pairs:
            pairs_by_output[output_name] = pairs
    return pairs_by_output


def percentage_change_histogram(numerator, denominator, weights):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        changes = numerator / denominator - 1.0

    valid = np.isfinite(changes) & np.isfinite(weights)
    if not np.any(valid):
        return np.zeros(len(BIN_EDGES) - 1, dtype=float)

    clipped_changes = np.clip(changes[valid], BIN_MIN, BIN_MAX)
    counts, _ = np.histogram(
        clipped_changes,
        bins=BIN_EDGES,
        weights=weights[valid],
    )
    return counts


def calculate_histograms(path, chunk_size):
    columns = pd.read_csv(path, nrows=0).columns
    if "w" not in columns:
        raise ValueError(f"Required weight column 'w' not found in {path}")

    pairs_by_output = adjacent_seed_pairs(columns)
    if not pairs_by_output:
        raise ValueError(f"Fewer than two compatible prediction columns found in {path}")

    histogram_counts = {
        output_name: {
            result_name: np.zeros(len(BIN_EDGES) - 1, dtype=float)
            for result_name, _, _ in pairs
        }
        for output_name, pairs in pairs_by_output.items()
    }
    prediction_columns = {
        column
        for pairs in pairs_by_output.values()
        for _, numerator, denominator in pairs
        for column in (numerator, denominator)
    }
    usecols = ["w", *sorted(prediction_columns)]

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size):
        weights = chunk["w"].to_numpy(dtype=float)
        for output_name, pairs in pairs_by_output.items():
            for result_name, numerator, denominator in pairs:
                histogram_counts[output_name][
                    result_name
                ] += percentage_change_histogram(
                    chunk[numerator].to_numpy(dtype=float),
                    chunk[denominator].to_numpy(dtype=float),
                    weights,
                )

    return histogram_counts


def histogram_frame(histogram_counts):
    frames = []
    is_multiclass = any(output_name != "" for output_name in histogram_counts)
    for output_name, output_counts in histogram_counts.items():
        data = {
            "bin_lower": BIN_EDGES[:-1],
            "bin_upper": BIN_EDGES[1:],
            **output_counts,
        }
        if is_multiclass:
            data = {"class": output_name, **data}
        frames.append(pd.DataFrame(data))

    frame = pd.concat(frames, ignore_index=True)
    # Avoid floating-point representations such as 0.09999999999999998.
    frame["bin_lower"] = frame["bin_lower"].round(2)
    frame["bin_upper"] = frame["bin_upper"].round(2)
    return frame


def output_path_for(input_path):
    return input_path.with_name(input_path.name[: -len(INPUT_SUFFIX)] + OUTPUT_SUFFIX)


def process_file(path, chunk_size):
    output_path = output_path_for(path)
    counts = calculate_histograms(path, chunk_size)
    histogram_frame(counts).to_csv(output_path, index=False)
    print(f"Saved {output_path}")
    return output_path


def main():
    args = parse_args()
    input_files = discover_input_files(args.prediction_dir, args.only_dataset)
    if not input_files:
        print("No prediction CSV files matched the requested inputs.")
        return

    for path in input_files:
        process_file(path, args.chunk_size)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compute decile summaries from paired smoothed/unsmoothed predictions.

Deciles are sorted in ascending order of the unsmoothed prediction, so decile 1
contains the lowest predictions.  French motor claims rows use ``w`` (policy
exposure) both to form approximately equal-exposure buckets and to calculate
the three averages.  All other datasets use equal row weights.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


PREDICTIONS_DIR = Path("/home/kerith/workspaces/smooth_multi/predictions")
OUTPUT_PATH = Path("/home/kerith/workspaces/smooth_multi/decile/decile_summary.csv")
FRENCH_MOTOR_DATASET = "French-Motor-Claims-Datasets-freMTPL2freq"
FILE_PATTERN = re.compile(
    r"^(?P<dataset>.+)_(?P<variant>smoothed|unsmoothed)_depth_"
    r"(?P<depth>\d+)_predictions\.csv$"
)
SCALAR_PREDICTION_PATTERN = re.compile(r"^pred_seed_(\d+)$")
MULTICLASS_PREDICTION_PATTERN = re.compile(r"^pred_seed_(\d+)_class_(.+)$")
OUTPUT_COLUMNS = [
    "dataset",
    "depth",
    "random_seed",
    "class",
    "decile_bucket",
    "average_actuals",
    "average_unsmoothed_predictions",
    "average_smoothed_predictions",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate decile summaries from saved CatBoost predictions."
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=PREDICTIONS_DIR,
        help="Directory containing paired *_predictions.csv files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_PATH,
        help="CSV file to which the combined decile summary is written.",
    )
    return parser.parse_args()


def discover_prediction_pairs(predictions_dir: Path) -> dict[tuple[str, int], dict[str, Path]]:
    if not predictions_dir.is_dir():
        raise FileNotFoundError(f"Predictions directory does not exist: {predictions_dir}")

    discovered: dict[tuple[str, int], dict[str, Path]] = defaultdict(dict)
    for path in sorted(predictions_dir.glob("*_predictions.csv")):
        match = FILE_PATTERN.fullmatch(path.name)
        if match is None:
            print(f"Skipping unrecognised predictions filename: {path.name}", flush=True)
            continue
        key = (match.group("dataset"), int(match.group("depth")))
        variant = match.group("variant")
        if variant in discovered[key]:
            raise ValueError(f"Duplicate {variant} predictions for dataset/depth {key}")
        discovered[key][variant] = path

    incomplete = {
        key: sorted({"smoothed", "unsmoothed"} - set(paths))
        for key, paths in discovered.items()
        if set(paths) != {"smoothed", "unsmoothed"}
    }
    if incomplete:
        details = "; ".join(f"{key}: missing {missing}" for key, missing in incomplete.items())
        raise ValueError(f"Every dataset/depth needs a prediction pair. {details}")
    if not discovered:
        raise FileNotFoundError(f"No paired *_predictions.csv files found in {predictions_dir}")
    return dict(discovered)


def safe_class_name(value: object) -> str:
    """Match the filename-safe class labels used by openml_prediction_utils.py."""
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    value = re.sub(r"_+", "_", value).strip("_.")
    return value[:120] or "dataset"


def align_prediction_frames(
    unsmoothed: pd.DataFrame, smoothed: pd.DataFrame, context: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"row_id", "target", "w"}
    for variant, frame in (("unsmoothed", unsmoothed), ("smoothed", smoothed)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{context} {variant} file is missing columns: {sorted(missing)}")
        if frame["row_id"].duplicated().any():
            raise ValueError(f"{context} {variant} file has duplicate row_id values")

    unsmoothed = unsmoothed.set_index("row_id", drop=False)
    smoothed = smoothed.set_index("row_id", drop=False)
    if set(unsmoothed.index) != set(smoothed.index):
        raise ValueError(f"{context} files do not contain the same row_id values")
    smoothed = smoothed.loc[unsmoothed.index]

    if not unsmoothed["target"].equals(smoothed["target"]):
        raise ValueError(f"{context} target values differ between the paired files")
    left_weights = pd.to_numeric(unsmoothed["w"], errors="coerce").to_numpy()
    right_weights = pd.to_numeric(smoothed["w"], errors="coerce").to_numpy()
    if not np.allclose(left_weights, right_weights, equal_nan=True):
        raise ValueError(f"{context} weights differ between the paired files")
    return unsmoothed, smoothed


def prediction_columns(frame: pd.DataFrame) -> dict[tuple[int, str | None], str]:
    """Map (seed, class token) to each probability/prediction column."""
    result: dict[tuple[int, str | None], str] = {}
    for column in frame.columns:
        scalar_match = SCALAR_PREDICTION_PATTERN.fullmatch(column)
        if scalar_match:
            result[(int(scalar_match.group(1)), None)] = column
            continue
        multiclass_match = MULTICLASS_PREDICTION_PATTERN.fullmatch(column)
        if multiclass_match:
            result[(int(multiclass_match.group(1)), multiclass_match.group(2))] = column
    return result


def binary_class_label(frame: pd.DataFrame, seed: int, context: str) -> str | None:
    column = f"pred_seed_{seed}_class"
    if column not in frame:
        return None
    labels = frame[column].dropna().astype(str).unique()
    if len(labels) != 1:
        raise ValueError(f"{context} column {column} must contain exactly one class label")
    return str(labels[0])


def numeric_values(series: pd.Series, label: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains missing or non-finite values")
    return values


def decile_buckets(sort_values: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """Return buckets 1-10, using stable sorting to make ties deterministic."""
    if len(sort_values) == 0:
        raise ValueError("Cannot calculate deciles for an empty prediction file")
    order = np.argsort(sort_values, kind="mergesort")
    sorted_buckets: np.ndarray
    if weights is None:
        sorted_buckets = np.minimum(np.arange(len(order)) * 10 // len(order), 9)
    else:
        sorted_weights = weights[order]
        total_weight = float(sorted_weights.sum())
        if not np.isfinite(total_weight) or total_weight <= 0:
            raise ValueError("French motor claim policy weights must have a positive sum")
        # A row is assigned according to the midpoint of its exposure interval.
        cumulative_midpoints = np.cumsum(sorted_weights) - 0.5 * sorted_weights
        sorted_buckets = np.floor(cumulative_midpoints * 10 / total_weight).astype(int)
        sorted_buckets = np.clip(sorted_buckets, 0, 9)

    buckets = np.empty(len(order), dtype=np.int8)
    buckets[order] = sorted_buckets + 1
    return buckets


def average(values: np.ndarray, weights: np.ndarray | None) -> float:
    if len(values) == 0:
        return np.nan
    if weights is None:
        return float(np.mean(values))
    weight_sum = float(weights.sum())
    return np.nan if weight_sum <= 0 else float(np.average(values, weights=weights))


def actuals_and_class_label(
    target: pd.Series,
    class_token: str | None,
    binary_label: str | None,
) -> tuple[np.ndarray, str]:
    if class_token is not None:
        target_tokens = target.map(safe_class_name)
        matching_targets = target.loc[target_tokens == class_token].astype(str).unique()
        output_label = str(matching_targets[0]) if len(matching_targets) == 1 else class_token
        return (target_tokens == class_token).to_numpy(dtype=float), output_label
    if binary_label is not None:
        return (target.astype(str) == binary_label).to_numpy(dtype=float), binary_label
    return numeric_values(target, "target"), ""


def summarize_pair(
    dataset: str,
    depth: int,
    paths: dict[str, Path],
) -> list[dict[str, object]]:
    context = f"dataset={dataset}, depth={depth}"
    unsmoothed, smoothed = align_prediction_frames(
        pd.read_csv(paths["unsmoothed"]),
        pd.read_csv(paths["smoothed"]),
        context,
    )
    unsmoothed_columns = prediction_columns(unsmoothed)
    smoothed_columns = prediction_columns(smoothed)
    if not unsmoothed_columns:
        raise ValueError(f"{context} contains no recognised prediction columns")
    if set(unsmoothed_columns) != set(smoothed_columns):
        raise ValueError(f"{context} files do not contain the same seed/class predictions")

    weights = None
    if dataset == FRENCH_MOTOR_DATASET:
        weights = numeric_values(unsmoothed["w"], f"{context} policy weights")
        if (weights < 0).any():
            raise ValueError(f"{context} policy weights cannot be negative")

    rows: list[dict[str, object]] = []
    for seed, class_token in sorted(
        unsmoothed_columns, key=lambda item: (item[0], "" if item[1] is None else item[1])
    ):
        unsmoothed_column = unsmoothed_columns[(seed, class_token)]
        smoothed_column = smoothed_columns[(seed, class_token)]
        unsmoothed_values = numeric_values(unsmoothed[unsmoothed_column], unsmoothed_column)
        smoothed_values = numeric_values(smoothed[smoothed_column], smoothed_column)

        unsmoothed_binary_label = binary_class_label(unsmoothed, seed, context)
        smoothed_binary_label = binary_class_label(smoothed, seed, context)
        if unsmoothed_binary_label != smoothed_binary_label:
            raise ValueError(f"{context}, seed={seed} has different binary class labels")
        actuals, output_class = actuals_and_class_label(
            unsmoothed["target"], class_token, unsmoothed_binary_label
        )
        buckets = decile_buckets(unsmoothed_values, weights)

        for bucket in range(1, 11):
            mask = buckets == bucket
            bucket_weights = None if weights is None else weights[mask]
            rows.append(
                {
                    "dataset": dataset,
                    "depth": depth,
                    "random_seed": seed,
                    "class": output_class,
                    "decile_bucket": bucket,
                    "average_actuals": average(actuals[mask], bucket_weights),
                    "average_unsmoothed_predictions": average(
                        unsmoothed_values[mask], bucket_weights
                    ),
                    "average_smoothed_predictions": average(
                        smoothed_values[mask], bucket_weights
                    ),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    pairs = discover_prediction_pairs(args.predictions_dir)
    datasets = sorted({dataset for dataset, _ in pairs})
    all_rows: list[dict[str, object]] = []

    print(
        f"Found {len(datasets)} datasets and {len(pairs)} paired dataset/depth files.",
        flush=True,
    )
    for dataset_number, dataset in enumerate(datasets, start=1):
        depths = sorted(depth for candidate, depth in pairs if candidate == dataset)
        print(
            f"[{dataset_number}/{len(datasets)}] Working on dataset: {dataset}",
            flush=True,
        )
        if dataset == FRENCH_MOTOR_DATASET:
            print("  Using policy-length weights for buckets and averages.", flush=True)
        for depth in depths:
            print(f"  Calculating depth {depth}", flush=True)
            all_rows.extend(summarize_pair(dataset, depth, pairs[(dataset, depth)]))

    summary = pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_path, index=False)
    print(f"Saved {len(summary):,} decile rows to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()

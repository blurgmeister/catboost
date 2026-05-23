#!/usr/bin/env python3
"""
Compute average split-point gaps for each dataset/depth/feature PDP row.

This retrains the same deterministic CatBoost models used for PDP generation,
extracts the actual split thresholds used in the trained trees, and writes one
row per dataset/depth/feature to a CSV in smooth_multi/.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from catboost import Pool
from sklearn.model_selection import train_test_split

from generate_pdps import (
    BASE_DIR,
    MANIFEST_PATH,
    OUTPUT_DIR,
    RANDOM_SEED,
    TARGET_COLUMN,
    build_model,
    cast_integer_columns_to_float,
    get_cat_feature_names,
    infer_task_type,
    prepare_model_data,
    safe_name,
)


PDP_SUMMARY_PATH = OUTPUT_DIR / "pdp_summary.csv"
RESULT_PATH = BASE_DIR / "avg_split_gaps.csv"


def parse_split(split_text: str) -> tuple[str, float] | None:
    if ", bin=" not in split_text:
        return None
    feature_part, border_part = split_text.split(", bin=", 1)
    return feature_part.strip(), float(border_part.strip())


def average_gap(values: list[float]) -> float | None:
    unique_sorted = sorted(set(values))
    if len(unique_sorted) < 2:
        return None
    gaps = [right - left for left, right in zip(unique_sorted, unique_sorted[1:])]
    return sum(gaps) / len(gaps)


def main() -> int:
    manifest = pd.read_csv(MANIFEST_PATH)
    pdp_summary = pd.read_csv(PDP_SUMMARY_PATH)
    pdp_summary_features = pdp_summary[["dataset_name", "depth", "feature"]].drop_duplicates()
    dataset_rows = []

    for _, manifest_row in manifest.iterrows():
        dataset_name = str(manifest_row["dataset_name"])
        suite_name = str(manifest_row["suite"])
        task_type = infer_task_type(suite_name)
        csv_path = Path(manifest_row["csv_path"])
        frame = pd.read_csv(csv_path)
        frame = frame.loc[frame[TARGET_COLUMN].notna()].reset_index(drop=True)
        X, y, row_weight, target_description = prepare_model_data(frame, suite_name)
        X = cast_integer_columns_to_float(X, exclude=set())
        cat_features = get_cat_feature_names(X)
        for column in cat_features:
            X[column] = X[column].fillna("__nan__").astype(str)

        if task_type == "classification":
            stratify = y if y.nunique(dropna=False) > 1 else None
            if row_weight is None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X,
                    y,
                    test_size=0.3,
                    random_state=RANDOM_SEED,
                    stratify=stratify,
                )
                weight_train = None
                weight_val = None
            else:
                split_values = train_test_split(
                    X,
                    y,
                    row_weight,
                    test_size=0.3,
                    random_state=RANDOM_SEED,
                    stratify=stratify,
                )
                X_train, X_val, y_train, y_val, weight_train, weight_val = split_values
            class_count = int(y.nunique(dropna=False))
        else:
            if row_weight is None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X,
                    y,
                    test_size=0.3,
                    random_state=RANDOM_SEED,
                )
                weight_train = None
                weight_val = None
            else:
                split_values = train_test_split(
                    X,
                    y,
                    row_weight,
                    test_size=0.3,
                    random_state=RANDOM_SEED,
                )
                X_train, X_val, y_train, y_val, weight_train, weight_val = split_values
            class_count = None

        dataset_summary = pdp_summary_features[pdp_summary_features["dataset_name"] == dataset_name]
        for depth in sorted(dataset_summary["depth"].unique()):
            print(f"Processing {dataset_name} depth={depth}")
            if row_weight is not None:
                print(f"  Using target={target_description} with exposure row weights")
            train_dir = OUTPUT_DIR / safe_name(dataset_name) / f"depth_{int(depth)}" / "catboost_info_gap_calc"
            train_dir.mkdir(parents=True, exist_ok=True)
            model = build_model(
                task_type,
                int(depth),
                cat_features,
                train_dir,
                class_count=class_count,
            )
            train_pool = Pool(X_train, y_train, cat_features=cat_features, weight=weight_train)
            eval_pool = Pool(X_val, y_val, cat_features=cat_features, weight=weight_val)
            model.fit(train_pool, eval_set=eval_pool, early_stopping_rounds=25, verbose=False)

            split_map: dict[str, list[float]] = {}
            for tree_idx in range(model.tree_count_):
                for split_text in model._get_tree_splits(tree_idx, train_pool):
                    parsed = parse_split(split_text)
                    if parsed is None:
                        continue
                    feature_name, border = parsed
                    split_map.setdefault(feature_name, []).append(border)

            dataset_depth_rows = dataset_summary[dataset_summary["depth"] == depth]
            for _, pdp_row in dataset_depth_rows.iterrows():
                feature = str(pdp_row["feature"])
                gap = average_gap(split_map.get(feature, []))
                dataset_rows.append(
                    {
                        "dataset": dataset_name,
                        "depth": int(depth),
                        "feature": feature,
                        "avg_dist": "" if gap is None else gap,
                    }
                )

    with RESULT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "depth", "feature", "avg_dist"])
        writer.writeheader()
        writer.writerows(dataset_rows)

    print(f"Wrote {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

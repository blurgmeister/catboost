#!/usr/bin/env python3
"""
Train CatBoost models for downloaded OpenML datasets and save PDP plots.

For every dataset in manifest.csv:
  - train one model at depth 1
  - train one model at depth 2
  - identify used features from feature importance
  - generate and save one PDP/ICE PNG per used feature

Outputs are written under /home/kerith/workspaces/smooth_multi/PDP.
"""

from __future__ import annotations

import csv
import os
import re
import argparse
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.inspection import partial_dependence
from sklearn.model_selection import train_test_split


BASE_DIR = Path("/home/kerith/workspaces/smooth_multi")
MANIFEST_PATH = BASE_DIR / "manifest.csv"
OUTPUT_DIR = BASE_DIR / "PDP"
TARGET_COLUMN = "__target__"
FRENCH_MOTOR_CLAIMS_SUITE = "OpenML-French-Motor-Claims"
FRENCH_MOTOR_CLAIMS_EXPOSURE_COLUMN = "Exposure"
CLASSIFICATION_SUITES = {"OpenML-CC18", "OpenML-Covertype"}
DEFAULT_MULTICLASS_CLASSES_BY_SUITE = {
    "OpenML-Covertype": ("1", "2"),
}
RANDOM_SEED = 42
DEPTHS = (1, 2)
MAX_ICE_TRACES = 100
MAX_PD_ROWS = 5000
GRID_RESOLUTION = 50
DEFAULT_ONLY_SUITE = "all"


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    value = re.sub(r"_+", "_", value).strip("_.")
    return value[:120] or "dataset"


def infer_task_type(suite_name: str) -> str:
    if suite_name in CLASSIFICATION_SUITES:
        return "classification"
    if suite_name in {"OpenML-CTR23", FRENCH_MOTOR_CLAIMS_SUITE}:
        return "regression"
    raise ValueError(f"Unsupported suite: {suite_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CatBoost models and save PDP plots for datasets in manifest.csv."
    )
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--only-suite",
        default=DEFAULT_ONLY_SUITE,
        help=(
            "Only process datasets from this suite. Defaults to all suites."
        ),
    )
    parser.add_argument(
        "--only-dataset-id",
        default=None,
        help="Only process a specific OpenML dataset id.",
    )
    parser.add_argument(
        "--only-dataset-name",
        default=None,
        help="Only process datasets whose name contains this case-insensitive text.",
    )
    parser.add_argument(
        "--pdp-class",
        action="append",
        default=None,
        help=(
            "For multiclass classification, output PDPs for this class label. "
            "May be passed multiple times. Covertype defaults to classes 1 and 2."
        ),
    )
    return parser.parse_args()


def filter_manifest(manifest: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    filtered = manifest
    if args.only_suite and args.only_suite.lower() != "all":
        filtered = filtered[filtered["suite"].astype(str) == args.only_suite]
    if args.only_dataset_id is not None:
        filtered = filtered[filtered["dataset_id"].astype(str) == str(args.only_dataset_id)]
    if args.only_dataset_name:
        pattern = re.escape(args.only_dataset_name)
        filtered = filtered[
            filtered["dataset_name"].astype(str).str.contains(pattern, case=False, regex=True, na=False)
        ]
    return filtered.reset_index(drop=True)


def get_cat_feature_names(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column])
        or pd.api.types.is_bool_dtype(frame[column])
    ]


def cast_integer_columns_to_float(frame: pd.DataFrame, exclude: set[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if column in exclude:
            continue
        if pd.api.types.is_integer_dtype(result[column]):
            result[column] = result[column].astype(float)
    return result


def prepare_model_data(
    frame: pd.DataFrame,
    suite_name: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None, str]:
    if suite_name != FRENCH_MOTOR_CLAIMS_SUITE:
        X = frame.drop(columns=[TARGET_COLUMN])
        y = frame[TARGET_COLUMN]
        return X, y, None, TARGET_COLUMN

    if FRENCH_MOTOR_CLAIMS_EXPOSURE_COLUMN not in frame.columns:
        raise ValueError(
            f"{FRENCH_MOTOR_CLAIMS_SUITE} requires "
            f"{FRENCH_MOTOR_CLAIMS_EXPOSURE_COLUMN!r} for row weights."
        )

    exposure = pd.to_numeric(frame[FRENCH_MOTOR_CLAIMS_EXPOSURE_COLUMN], errors="coerce")
    claim_count = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    valid = claim_count.notna() & exposure.notna() & (exposure > 0)
    dropped = len(frame) - int(valid.sum())
    if dropped:
        print(f"  Dropping {dropped} rows with missing target or non-positive exposure")

    filtered = frame.loc[valid].reset_index(drop=True)
    exposure = exposure.loc[valid].reset_index(drop=True).astype(float)
    claim_count = claim_count.loc[valid].reset_index(drop=True).astype(float)
    X = filtered.drop(columns=[TARGET_COLUMN, FRENCH_MOTOR_CLAIMS_EXPOSURE_COLUMN])
    y = claim_count / exposure
    return X, y, exposure, "claim_count_per_exposure"


def sample_for_pd(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) <= MAX_PD_ROWS:
        return X, y
    if y.nunique(dropna=False) > 1 and len(y) > MAX_PD_ROWS:
        try:
            _, X_sample, _, y_sample = train_test_split(
                X,
                y,
                test_size=MAX_PD_ROWS,
                random_state=RANDOM_SEED,
                stratify=y if y.nunique() <= 50 else None,
            )
            return X_sample.reset_index(drop=True), y_sample.reset_index(drop=True)
        except ValueError:
            pass
    sample = X.sample(n=MAX_PD_ROWS, random_state=RANDOM_SEED)
    return sample.reset_index(drop=True), y.loc[sample.index].reset_index(drop=True)


def build_model(
    task_type: str,
    depth: int,
    cat_features: list[str],
    train_dir: Path,
    class_count: int | None = None,
):
    base_params = {
        "depth": depth,
        "iterations": 400,
        "learning_rate": 0.05,
        "random_seed": RANDOM_SEED,
        "verbose": False,
        "train_dir": str(train_dir),
        "allow_writing_files": True,
        "thread_count": -1,
    }
    if task_type == "classification":
        loss_function = "Logloss" if (class_count or 0) <= 2 else "MultiClass"
        return CatBoostClassifier(
            loss_function=loss_function,
            eval_metric=loss_function,
            cat_features=cat_features,
            **base_params,
        )
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        cat_features=cat_features,
        **base_params,
    )


def get_used_features(model, feature_names: list[str]) -> list[str]:
    importances = np.asarray(model.get_feature_importance(), dtype=float)
    used = [name for name, importance in zip(feature_names, importances) if importance > 0]
    if used:
        return used
    order = np.argsort(importances)[::-1]
    fallback = [feature_names[idx] for idx in order[: min(10, len(feature_names))]]
    return fallback


def normalize_pd_output(
    pd_results: dict,
    class_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_values = np.asarray(pd_results["grid_values"][0])
    ice_raw = np.asarray(pd_results["individual"])
    pdp_raw = np.asarray(pd_results["average"])

    if class_index is not None:
        if ice_raw.ndim != 3 or pdp_raw.ndim != 2:
            raise ValueError(
                "Expected multiclass PDP output with individual shape "
                "(n_classes, n_samples, n_grid) and average shape (n_classes, n_grid)."
            )
        ice_values = ice_raw[class_index]
        pdp_values = pdp_raw[class_index]
    elif ice_raw.ndim == 3:
        ice_values = ice_raw[0]
    else:
        ice_values = ice_raw

    if class_index is None and pdp_raw.ndim == 2:
        pdp_values = pdp_raw[0]
    elif class_index is None:
        pdp_values = pdp_raw

    return grid_values, ice_values, pdp_values


def get_multiclass_class_indexes(
    model,
    suite_name: str,
    requested_classes: list[str] | None,
) -> list[tuple[str, int]]:
    classes = [str(value) for value in model.classes_]
    selected_classes = requested_classes or list(DEFAULT_MULTICLASS_CLASSES_BY_SUITE.get(suite_name, classes))
    selected: list[tuple[str, int]] = []
    for class_label in selected_classes:
        if class_label not in classes:
            raise ValueError(
                f"Requested PDP class {class_label!r} is not in model classes {classes}."
            )
        selected.append((class_label, classes.index(class_label)))
    return selected


def save_pdp_plot(
    dataset_name: str,
    depth: int,
    feature: str,
    task_type: str,
    grid_values: np.ndarray,
    ice_values: np.ndarray,
    pdp_values: np.ndarray,
    output_path: Path,
    class_label: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    trace_count = min(MAX_ICE_TRACES, ice_values.shape[0])
    if trace_count > 0:
        rng = np.random.default_rng(RANDOM_SEED)
        indexes = rng.choice(ice_values.shape[0], size=trace_count, replace=False)
        for idx in indexes:
            ax.plot(grid_values, ice_values[idx], color="gray", alpha=0.12, linewidth=0.6)

    ax.plot(grid_values, pdp_values, color="blue", linewidth=2, label="PDP")
    class_suffix = "" if class_label is None else f" | class={class_label}"
    ax.set_title(f"{dataset_name} | depth={depth} | {feature}{class_suffix}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Prediction" if task_type == "regression" else "Probability")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_summary(rows: list[dict], output_dir: Path) -> None:
    if not rows:
        return
    summary_path = output_dir / "pdp_summary.csv"
    fieldnames = ["suite", "dataset_name", "task_type", "depth", "feature", "target_class", "plot_path"]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = filter_manifest(pd.read_csv(args.manifest_path), args)
    if manifest.empty:
        print("No datasets matched the requested filters.")
        return 0

    summary_rows: list[dict] = []

    for _, row in manifest.iterrows():
        dataset_name = str(row["dataset_name"])
        suite_name = str(row["suite"])
        csv_path = Path(row["csv_path"])
        task_type = infer_task_type(suite_name)
        dataset_dir = args.output_dir / safe_name(dataset_name)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {dataset_name} ({task_type})")
        frame = pd.read_csv(csv_path)
        frame = frame.loc[frame[TARGET_COLUMN].notna()].reset_index(drop=True)
        X, y, row_weight, target_description = prepare_model_data(frame, suite_name)
        X = cast_integer_columns_to_float(X, exclude=set())
        class_count = int(y.nunique(dropna=False)) if task_type == "classification" else None

        if row_weight is not None:
            print(
                f"  Using target={target_description}, "
                f"row_weight={FRENCH_MOTOR_CLAIMS_EXPOSURE_COLUMN}, "
                f"excluded_feature={FRENCH_MOTOR_CLAIMS_EXPOSURE_COLUMN}"
            )

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

        X_pd, y_pd = sample_for_pd(X, y)

        for depth in DEPTHS:
            depth_dir = dataset_dir / f"depth_{depth}"
            depth_dir.mkdir(parents=True, exist_ok=True)
            train_dir = depth_dir / "catboost_info"
            train_dir.mkdir(parents=True, exist_ok=True)

            print(f"  Training depth={depth}")
            model = build_model(task_type, depth, cat_features, train_dir, class_count=class_count)
            train_pool = Pool(X_train, y_train, cat_features=cat_features, weight=weight_train)
            eval_pool = Pool(X_val, y_val, cat_features=cat_features, weight=weight_val)
            model.fit(
                train_pool,
                eval_set=eval_pool,
                early_stopping_rounds=25,
                verbose=False,
            )

            used_features = get_used_features(model, list(X.columns))
            if not used_features:
                print("    No used features found, skipping PDP generation")
                continue

            multiclass_class_indexes = (
                get_multiclass_class_indexes(model, suite_name, args.pdp_class)
                if task_type == "classification" and class_count and class_count > 2
                else []
            )

            for feature in used_features:
                print(f"    PDP for {feature}")
                pd_results = partial_dependence(
                    model,
                    X_pd,
                    [feature],
                    kind="both",
                    grid_resolution=GRID_RESOLUTION,
                    response_method="auto",
                )
                if multiclass_class_indexes:
                    for class_label, class_index in multiclass_class_indexes:
                        grid_values, ice_values, pdp_values = normalize_pd_output(
                            pd_results,
                            class_index=class_index,
                        )
                        output_path = depth_dir / f"{safe_name(feature)}_class-{safe_name(class_label)}_pdp.png"
                        save_pdp_plot(
                            dataset_name=dataset_name,
                            depth=depth,
                            feature=feature,
                            task_type=task_type,
                            grid_values=grid_values,
                            ice_values=ice_values,
                            pdp_values=pdp_values,
                            output_path=output_path,
                            class_label=class_label,
                        )
                        summary_rows.append(
                            {
                                "suite": suite_name,
                                "dataset_name": dataset_name,
                                "task_type": task_type,
                                "depth": depth,
                                "feature": feature,
                                "target_class": class_label,
                                "plot_path": str(output_path),
                            }
                        )
                else:
                    grid_values, ice_values, pdp_values = normalize_pd_output(pd_results)
                    output_path = depth_dir / f"{safe_name(feature)}_pdp.png"
                    save_pdp_plot(
                        dataset_name=dataset_name,
                        depth=depth,
                        feature=feature,
                        task_type=task_type,
                        grid_values=grid_values,
                        ice_values=ice_values,
                        pdp_values=pdp_values,
                        output_path=output_path,
                    )
                    summary_rows.append(
                        {
                            "suite": suite_name,
                            "dataset_name": dataset_name,
                            "task_type": task_type,
                            "depth": depth,
                            "feature": feature,
                            "target_class": "",
                            "plot_path": str(output_path),
                        }
                    )

    write_summary(summary_rows, args.output_dir)
    print(f"Wrote PDP plots under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

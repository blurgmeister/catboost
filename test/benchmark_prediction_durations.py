#!/usr/bin/env python3
"""
Benchmark whole-dataset prediction duration across CatBoost, XGBoost, and LightGBM.

This script is modeled after:
  - /home/kerith/workspaces/catboost/test/measure_smoothing_prediction_times.py
  - /home/kerith/workspaces/catboost/test/measure_other_prediction_times.py

It also follows /home/kerith/workspaces/smooth_multi/generate_pdps.py for
dataset loading, target cleaning, task-type inference, and categorical handling.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import re
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from generate_pdps import (
    BASE_DIR,
    FRENCH_MOTOR_CLAIMS_EXPOSURE_COLUMN,
    FRENCH_MOTOR_CLAIMS_SUITE,
    MANIFEST_PATH,
    RANDOM_SEED,
    TARGET_COLUMN,
    cast_integer_columns_to_float,
    get_cat_feature_names,
    infer_task_type,
)


AVG_GAPS_PATH = BASE_DIR / "avg_split_gaps.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "prediction_duration_benchmark.csv"
DEFAULT_START_SEED = 17
DEFAULT_MODELS_PER_DEPTH = 25
DEFAULT_DEPTHS = [1, 2, 3, 4, 5, 6]
TEST_SIZE = 0.3
N_ESTIMATORS = 1000
EARLY_STOPPING_ROUNDS = 25

FIELDNAMES = [
    "model_type",
    "dataset",
    "depth",
    "n_rows",
    "n_features_used",
    "model_seed",
    "prediction_duration_seconds",
    "interpolation_enabled",
    "interpolation_spans_applied",
    "train_eval_metric",
    "validation_eval_metric",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark whole-dataset prediction time for CatBoost, XGBoost, and LightGBM."
    )
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--start-seed", type=int, default=DEFAULT_START_SEED)
    parser.add_argument("--models-per-depth", type=int, default=DEFAULT_MODELS_PER_DEPTH)
    parser.add_argument("--depths", type=int, nargs="+", default=DEFAULT_DEPTHS)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing CSV instead of replacing it.",
    )
    return parser.parse_args()


def model_seeds(start_seed: int, models_per_depth: int):
    return range(start_seed, start_seed + models_per_depth)


def load_gap_map() -> dict[tuple[str, int], dict[str, float]]:
    gap_frame = pd.read_csv(AVG_GAPS_PATH)
    span_column = "selected" if "selected" in gap_frame.columns else "avg_dist"
    gap_frame["span"] = pd.to_numeric(gap_frame[span_column], errors="coerce")
    if gap_frame["span"].isna().all():
        gap_frame["span"] = pd.to_numeric(gap_frame["avg_dist"], errors="coerce")
    gap_frame = gap_frame.loc[gap_frame["span"].notna() & (gap_frame["span"] > 0)].copy()
    gap_map: dict[tuple[str, int], dict[str, float]] = {}
    for _, row in gap_frame.iterrows():
        key = (str(row["dataset"]), int(row["depth"]))
        gap_map.setdefault(key, {})[str(row["feature"])] = float(row["span"])
    return gap_map


def get_interpolation_spans(
    gap_map: dict[tuple[str, int], dict[str, float]],
    dataset_name: str,
    depth: int,
) -> dict[str, float]:
    spans = gap_map.get((dataset_name, depth), {})
    if spans:
        return spans
    if depth > 2:
        return gap_map.get((dataset_name, 2), {})
    return {}


def map_spans_to_model_features(
    interpolation_spans: dict[str, float],
    model_features: list[str],
) -> dict[str, float]:
    exact_features = set(model_features)
    sanitized_to_feature = {}
    for feature in model_features:
        sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(feature).strip())
        sanitized = re.sub(r"_+", "_", sanitized).strip("_.")
        sanitized_to_feature.setdefault(sanitized[:120] or "dataset", feature)

    mapped = {}
    for feature, span in interpolation_spans.items():
        if feature in exact_features:
            mapped[feature] = span
        elif feature in sanitized_to_feature:
            mapped[sanitized_to_feature[feature]] = span
    return mapped


def load_dataset(row: pd.Series) -> tuple[pd.DataFrame, pd.Series, str, list[str], int | None, pd.Series | None]:
    suite_name = str(row["suite"])
    task_type = infer_task_type(suite_name)
    frame = pd.read_csv(row["csv_path"])
    frame = frame.loc[frame[TARGET_COLUMN].notna()].reset_index(drop=True)

    row_weight = None
    if suite_name == FRENCH_MOTOR_CLAIMS_SUITE:
        exposure = pd.to_numeric(frame[FRENCH_MOTOR_CLAIMS_EXPOSURE_COLUMN], errors="coerce")
        claim_count = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
        valid = claim_count.notna() & exposure.notna() & (exposure > 0)
        frame = frame.loc[valid].reset_index(drop=True)
        exposure = exposure.loc[valid].reset_index(drop=True).astype(float)
        claim_count = claim_count.loc[valid].reset_index(drop=True).astype(float)
        X = frame.drop(columns=[TARGET_COLUMN, FRENCH_MOTOR_CLAIMS_EXPOSURE_COLUMN])
        y = claim_count / exposure
        row_weight = exposure
    else:
        X = frame.drop(columns=[TARGET_COLUMN])
        y = frame[TARGET_COLUMN].copy()

    X = cast_integer_columns_to_float(X, exclude=set())
    cat_features = get_cat_feature_names(X)
    for column in cat_features:
        X[column] = X[column].fillna("__nan__").astype(str)

    class_count: int | None = None
    if task_type == "classification":
        y = pd.Series(pd.Categorical(y).codes, index=y.index)
        class_count = int(y.nunique(dropna=False))
    else:
        y = pd.to_numeric(y)

    return X, y, task_type, cat_features, class_count, row_weight


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    row_weight: pd.Series | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series | None, pd.Series | None]:
    stratify = None
    if task_type == "classification" and y.nunique(dropna=False) > 1:
        stratify = y
    if row_weight is None:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=stratify,
        )
        return X_train, X_val, y_train, y_val, None, None
    split_values = train_test_split(
        X,
        y,
        row_weight,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )
    X_train, X_val, y_train, y_val, weight_train, weight_val = split_values
    return X_train, X_val, y_train, y_val, weight_train, weight_val


def fit_ordinal_encoder(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_full: pd.DataFrame,
    cat_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not cat_features:
        return X_train.copy(), X_val.copy(), X_full.copy()

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )
    encoder.fit(X_train[cat_features])

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result[cat_features] = encoder.transform(frame[cat_features])
        return result

    return transform(X_train), transform(X_val), transform(X_full)


def get_used_feature_count(importances: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(importances, dtype=float) > 0))


def measure_prediction_duration(model, X) -> float:
    gc.collect()
    start = perf_counter()
    predictions = model.predict(X)
    duration = perf_counter() - start
    if len(predictions) != len(X):
        raise RuntimeError(f"Expected {len(X)} predictions, received {len(predictions)}")
    return duration


def build_catboost_model(
    task_type: str,
    depth: int,
    seed: int,
    cat_features: list[str],
    class_count: int | None,
    interpolation_enabled: bool,
    interpolation_spans: dict[str, float],
):
    common_params = {
        "depth": depth,
        "iterations": N_ESTIMATORS,
        "random_seed": seed,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
        "cat_features": cat_features,
    }

    if task_type == "classification":
        loss_function = "Logloss" if (class_count or 0) <= 2 else "MultiClass"
        model = CatBoostClassifier(
            loss_function=loss_function,
            eval_metric=loss_function,
            **common_params,
        )
    else:
        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            **common_params,
        )

    if interpolation_enabled and interpolation_spans:
        model.set_params(
            interpolation_enabled=True,
            interpolation_type="Sigmoid",
            interpolation_span_mode={feature: "Absolute" for feature in interpolation_spans},
            interpolation_span=interpolation_spans,
            interpolation_min_span=0.0,
        )
    return model


def build_xgboost_model(task_type: str, depth: int, seed: int, class_count: int | None):
    common_params = {
        "n_estimators": N_ESTIMATORS,
        "max_depth": depth,
        "random_state": seed,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "verbosity": 0,
    }

    if task_type == "classification":
        if (class_count or 0) <= 2:
            return xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                **common_params,
            )
        return xgb.XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=class_count,
            **common_params,
        )

    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        **common_params,
    )


def build_lightgbm_model(task_type: str, depth: int, seed: int, class_count: int | None):
    common_params = {
        "n_estimators": N_ESTIMATORS,
        "max_depth": depth,
        "num_leaves": 2 ** max(depth, 1),
        "random_state": seed,
        "verbosity": -1,
    }

    if task_type == "classification":
        if (class_count or 0) <= 2:
            return lgb.LGBMClassifier(objective="binary", **common_params)
        return lgb.LGBMClassifier(
            objective="multiclass",
            num_class=class_count,
            **common_params,
        )

    return lgb.LGBMRegressor(objective="regression", **common_params)


def fit_catboost_model(
    model,
    X_train,
    X_val,
    y_train,
    y_val,
    cat_features: list[str],
    weight_train=None,
    weight_val=None,
):
    train_pool = Pool(X_train, y_train, cat_features=cat_features, weight=weight_train)
    eval_pool = Pool(X_val, y_val, cat_features=cat_features, weight=weight_val)
    model.fit(
        train_pool,
        eval_set=eval_pool,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=False,
    )
    return train_pool, eval_pool


def fit_xgboost_model(model, X_train, X_val, y_train, y_val, weight_train=None, weight_val=None):
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        sample_weight=weight_train,
        sample_weight_eval_set=[weight_train, weight_val],
        verbose=False,
    )


def fit_lightgbm_model(model, X_train, X_val, y_train, y_val, weight_train=None, weight_val=None):
    model.fit(
        X_train,
        y_train,
        sample_weight=weight_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_sample_weight=[weight_train, weight_val],
        eval_names=["training", "validation"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )


def get_catboost_metrics(model) -> tuple[float | str, float | str]:
    evals = model.get_evals_result()
    metric_name = model.get_best_score().get("validation", {})
    if metric_name:
        metric_key = next(iter(metric_name.keys()))
    else:
        learn_metrics = evals.get("learn", {})
        metric_key = next(iter(learn_metrics.keys())) if learn_metrics else None

    if not metric_key:
        return "", ""

    learn_values = evals.get("learn", {}).get(metric_key, [])
    validation_values = evals.get("validation", {}).get(metric_key, [])
    train_metric = learn_values[-1] if learn_values else ""
    validation_metric = validation_values[-1] if validation_values else ""
    return train_metric, validation_metric


def get_xgboost_metrics(model) -> tuple[float | str, float | str]:
    evals = model.evals_result()
    train_entry = evals.get("validation_0", {})
    validation_entry = evals.get("validation_1", {})
    metric_key = next(iter(validation_entry.keys())) if validation_entry else None
    if not metric_key:
        return "", ""
    validation_values = validation_entry.get(metric_key, [])
    train_values = train_entry.get(metric_key, [])
    train_metric = train_values[-1] if train_values else ""
    validation_metric = validation_values[-1] if validation_values else ""
    return train_metric, validation_metric


def get_lightgbm_metrics(model) -> tuple[float | str, float | str]:
    results = getattr(model, "evals_result_", {})
    train_entry = results.get("training", {})
    validation_entry = results.get("validation", {})
    metric_key = next(iter(validation_entry.keys())) if validation_entry else None
    if not metric_key:
        return "", ""
    validation_values = validation_entry.get(metric_key, [])
    train_values = train_entry.get(metric_key, [])
    train_metric = train_values[-1] if train_values else ""
    validation_metric = validation_values[-1] if validation_values else ""
    return train_metric, validation_metric


def normalize_interpolation_value(value) -> str:
    if pd.isna(value) or value == "":
        return ""
    if isinstance(value, bool):
        return str(value)
    text = str(value).strip()
    if text in {"True", "False"}:
        return text
    if text in {"1", "1.0"}:
        return "True"
    if text in {"0", "0.0"}:
        return "False"
    return text


def completion_key(
    dataset_name: str,
    depth: int,
    seed: int,
    model_type: str,
    interpolation_enabled,
) -> tuple[str, int, int, str, str]:
    return (
        str(dataset_name),
        int(depth),
        int(seed),
        str(model_type),
        normalize_interpolation_value(interpolation_enabled),
    )


def load_completed_keys(output_path: Path) -> set[tuple[str, int, int, str, str]]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()

    existing = pd.read_csv(output_path)
    required_columns = {"dataset", "depth", "model_seed", "model_type", "interpolation_enabled"}
    if not required_columns.issubset(existing.columns):
        return set()

    completed = set()
    for _, row in existing.iterrows():
        completed.add(
            completion_key(
                row["dataset"],
                row["depth"],
                row["model_seed"],
                row["model_type"],
                row["interpolation_enabled"],
            )
        )
    return completed


def run_catboost_once(
    dataset_name: str,
    task_type: str,
    depth: int,
    seed: int,
    X: pd.DataFrame,
    y: pd.Series,
    cat_features: list[str],
    class_count: int | None,
    row_weight: pd.Series | None,
    interpolation_enabled: bool,
    interpolation_spans: dict[str, float],
) -> dict:
    X_train, X_val, y_train, y_val, weight_train, weight_val = split_dataset(
        X,
        y,
        task_type,
        row_weight,
    )
    interpolation_spans = map_spans_to_model_features(interpolation_spans, list(X.columns))
    interpolation_spans_applied = bool(interpolation_enabled and interpolation_spans)
    model = build_catboost_model(
        task_type=task_type,
        depth=depth,
        seed=seed,
        cat_features=cat_features,
        class_count=class_count,
        interpolation_enabled=interpolation_enabled,
        interpolation_spans=interpolation_spans,
    )
    fit_catboost_model(
        model,
        X_train,
        X_val,
        y_train,
        y_val,
        cat_features,
        weight_train=weight_train,
        weight_val=weight_val,
    )
    duration = measure_prediction_duration(model, X)
    n_features_used = get_used_feature_count(model.get_feature_importance())
    train_metric, validation_metric = get_catboost_metrics(model)
    return {
        "model_type": "catboost",
        "dataset": dataset_name,
        "depth": depth,
        "n_rows": len(X),
        "n_features_used": n_features_used,
        "model_seed": seed,
        "prediction_duration_seconds": duration,
        "interpolation_enabled": interpolation_enabled,
        "interpolation_spans_applied": interpolation_spans_applied,
        "train_eval_metric": train_metric,
        "validation_eval_metric": validation_metric,
    }


def run_other_once(
    model_type: str,
    dataset_name: str,
    task_type: str,
    depth: int,
    seed: int,
    X: pd.DataFrame,
    y: pd.Series,
    cat_features: list[str],
    class_count: int | None,
    row_weight: pd.Series | None,
) -> dict:
    X_train, X_val, y_train, y_val, weight_train, weight_val = split_dataset(
        X,
        y,
        task_type,
        row_weight,
    )
    X_train_enc, X_val_enc, X_full_enc = fit_ordinal_encoder(X_train, X_val, X, cat_features)

    if model_type == "xgboost":
        model = build_xgboost_model(task_type, depth, seed, class_count)
        fit_xgboost_model(
            model,
            X_train_enc,
            X_val_enc,
            y_train,
            y_val,
            weight_train=weight_train,
            weight_val=weight_val,
        )
        train_metric, validation_metric = get_xgboost_metrics(model)
    elif model_type == "lightgbm":
        model = build_lightgbm_model(task_type, depth, seed, class_count)
        fit_lightgbm_model(
            model,
            X_train_enc,
            X_val_enc,
            y_train,
            y_val,
            weight_train=weight_train,
            weight_val=weight_val,
        )
        train_metric, validation_metric = get_lightgbm_metrics(model)
    else:
        raise ValueError(f"Unsupported model_type={model_type}")

    duration = measure_prediction_duration(model, X_full_enc)
    n_features_used = get_used_feature_count(model.feature_importances_)
    return {
        "model_type": model_type,
        "dataset": dataset_name,
        "depth": depth,
        "n_rows": len(X),
        "n_features_used": n_features_used,
        "model_seed": seed,
        "prediction_duration_seconds": duration,
        "interpolation_enabled": "",
        "interpolation_spans_applied": "",
        "train_eval_metric": train_metric,
        "validation_eval_metric": validation_metric,
    }


def write_results(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    gap_map = load_gap_map()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    completed_keys = load_completed_keys(output_path) if file_exists else set()
    mode = "a" if file_exists and completed_keys else "w"

    with output_path.open(mode, newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES)
        if mode == "w":
            writer.writeheader()

        for _, manifest_row in manifest.iterrows():
            dataset_name = str(manifest_row["dataset_name"])
            print(f"Loading dataset={dataset_name}", flush=True)
            X, y, task_type, cat_features, class_count, row_weight = load_dataset(manifest_row)

            for depth in args.depths:
                interpolation_spans = get_interpolation_spans(gap_map, dataset_name, depth)
                catboost_interpolation_modes = [False, True]

                for seed in model_seeds(args.start_seed, args.models_per_depth):
                    for interpolation_enabled in catboost_interpolation_modes:
                        key = completion_key(
                            dataset_name,
                            depth,
                            seed,
                            "catboost",
                            interpolation_enabled,
                        )
                        if key in completed_keys:
                            print(
                                f"Skipping completed model_type=catboost dataset={dataset_name} "
                                f"depth={depth} seed={seed} interpolation_enabled={interpolation_enabled}",
                                flush=True,
                            )
                            continue
                        print(
                            f"Training model_type=catboost dataset={dataset_name} "
                            f"depth={depth} seed={seed} interpolation_enabled={interpolation_enabled}",
                            flush=True,
                        )
                        row = run_catboost_once(
                            dataset_name=dataset_name,
                            task_type=task_type,
                            depth=depth,
                            seed=seed,
                            X=X,
                            y=y,
                            cat_features=cat_features,
                            class_count=class_count,
                            row_weight=row_weight,
                            interpolation_enabled=interpolation_enabled,
                            interpolation_spans=interpolation_spans,
                        )
                        writer.writerow(row)
                        completed_keys.add(key)
                        output_file.flush()

                    for model_type in ("xgboost", "lightgbm"):
                        key = completion_key(
                            dataset_name,
                            depth,
                            seed,
                            model_type,
                            "",
                        )
                        if key in completed_keys:
                            print(
                                f"Skipping completed model_type={model_type} dataset={dataset_name} "
                                f"depth={depth} seed={seed}",
                                flush=True,
                            )
                            continue
                        print(
                            f"Training model_type={model_type} dataset={dataset_name} "
                            f"depth={depth} seed={seed}",
                            flush=True,
                        )
                        row = run_other_once(
                            model_type=model_type,
                            dataset_name=dataset_name,
                            task_type=task_type,
                            depth=depth,
                            seed=seed,
                            X=X,
                            y=y,
                            cat_features=cat_features,
                            class_count=class_count,
                            row_weight=row_weight,
                        )
                        writer.writerow(row)
                        completed_keys.add(key)
                        output_file.flush()


def main() -> None:
    args = parse_args()
    write_results(args)
    print(f"Saved {args.output_path}", flush=True)


if __name__ == "__main__":
    main()

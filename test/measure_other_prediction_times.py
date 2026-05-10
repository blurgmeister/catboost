import argparse
import csv
import gc
import os
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


OUTPUT_DIR = "/home/kerith/workspaces/smooth_simple"
DEFAULT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "prediction_timing_benchmark_other.csv")
SPLIT_SEED = 42
DEFAULT_START_SEED = 17
DEFAULT_MODELS_PER_DEPTH = 25
DEFAULT_DEPTHS = [1, 2, 3, 4, 5, 6]
EARLY_STOPPING_ROUNDS = 25
N_ESTIMATORS = 1000


TASKS = [
    {
        "name": "california_housing",
        "data_path": "/home/kerith/workspaces/test_output/regression/california_housing.csv",
        "task_type": "regression",
        "monotone_constraints": {
            "AveBedrms": 1,
            "AveOccup": -1,
            "HouseAge": 1,
            "Latitude": -1,
            "Longitude": -1,
            "MedInc": 1,
            "Population": 1,
        },
    },
    {
        "name": "breast_cancer",
        "data_path": "/home/kerith/workspaces/test_output/classification/breast_cancer.csv",
        "task_type": "classification",
        "monotone_constraints": {
            "mean concave points": -1,
            "mean fractal dimension": 1,
            "perimeter error": -1,
            "radius error": -1,
            "worst area": -1,
            "worst concave points": -1,
            "worst concavity": -1,
            "worst perimeter": -1,
            "worst radius": -1,
            "worst texture": -1,
            "mean concavity": -1,
        },
    },
]


FIELDNAMES = [
    "model_type",
    "dataset",
    "depth",
    "model_seed",
    "prediction_duration_seconds",
    "n_rows_scored",
    "n_features",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark whole-dataset XGBoost and LightGBM prediction time."
    )
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--start-seed", type=int, default=DEFAULT_START_SEED)
    parser.add_argument("--models-per-depth", type=int, default=DEFAULT_MODELS_PER_DEPTH)
    parser.add_argument("--depths", type=int, nargs="+", default=DEFAULT_DEPTHS)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing CSV instead of replacing it.",
    )
    return parser.parse_args()


def model_seeds(start_seed, models_per_depth):
    return range(start_seed, start_seed + models_per_depth)


def load_task_data(task):
    df = pd.read_csv(task["data_path"])
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def ordered_constraints(task, X):
    constraints = task["monotone_constraints"]
    return [constraints.get(feature_name, 0) for feature_name in X.columns]


def build_xgboost_model(task, X, depth, seed):
    common_params = {
        "n_estimators": N_ESTIMATORS,
        "max_depth": depth,
        "random_state": seed,
        "monotone_constraints": tuple(ordered_constraints(task, X)),
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    }

    if task["task_type"] == "classification":
        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            **common_params,
        )

    return xgb.XGBRegressor(
        objective="count:poisson",
        eval_metric="poisson-nloglik",
        **common_params,
    )


def build_lightgbm_model(task, X, depth, seed):
    common_params = {
        "n_estimators": N_ESTIMATORS,
        "max_depth": depth,
        "num_leaves": 2**depth,
        "random_state": seed,
        "monotone_constraints": ordered_constraints(task, X),
        "verbosity": -1,
    }

    if task["task_type"] == "classification":
        return lgb.LGBMClassifier(objective="binary", **common_params)

    return lgb.LGBMRegressor(objective="poisson", **common_params)


def build_model(model_type, task, X, depth, seed):
    if model_type == "xgboost":
        return build_xgboost_model(task, X, depth, seed)
    if model_type == "lightgbm":
        return build_lightgbm_model(task, X, depth, seed)
    raise ValueError(f"Unsupported model_type={model_type}")


def fit_model(model_type, model, X_train, X_val, y_train, y_val):
    if model_type == "lightgbm":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        return

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )


def measure_prediction_duration(model, X):
    gc.collect()
    start = perf_counter()
    predictions = model.predict(X)
    duration = perf_counter() - start

    if len(predictions) != len(X):
        raise RuntimeError(
            f"Expected {len(X)} predictions, received {len(predictions)}"
        )
    return duration


def run_one_model(model_type, task, X, y, depth, seed):
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=SPLIT_SEED,
    )

    model = build_model(model_type, task, X, depth, seed)
    fit_model(model_type, model, X_train, X_val, y_train, y_val)
    duration = measure_prediction_duration(model, X)

    return {
        "model_type": model_type,
        "dataset": task["name"],
        "depth": depth,
        "model_seed": seed,
        "prediction_duration_seconds": duration,
        "n_rows_scored": len(X),
        "n_features": len(X.columns),
    }


def write_results(args):
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    file_exists = os.path.exists(args.output_path)
    mode = "a" if args.append else "w"

    with open(args.output_path, mode, newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES)
        if not args.append or not file_exists:
            writer.writeheader()

        for task in TASKS:
            X, y = load_task_data(task)
            for model_type in ["xgboost", "lightgbm"]:
                for depth in args.depths:
                    for seed in model_seeds(args.start_seed, args.models_per_depth):
                        print(
                            "Training "
                            f"model_type={model_type} "
                            f"dataset={task['name']} "
                            f"depth={depth} "
                            f"seed={seed}",
                            flush=True,
                        )
                        row = run_one_model(model_type, task, X, y, depth, seed)
                        writer.writerow(row)
                        output_file.flush()
                        print(
                            "Recorded "
                            f"duration={row['prediction_duration_seconds']:.9f}s",
                            flush=True,
                        )


def main():
    args = parse_args()
    write_results(args)
    print(f"Saved {args.output_path}")


if __name__ == "__main__":
    main()

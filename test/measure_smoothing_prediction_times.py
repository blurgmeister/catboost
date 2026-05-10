import argparse
import copy
import csv
import gc
import os
from time import perf_counter

import catboost as cb
import pandas as pd
from sklearn.model_selection import train_test_split


OUTPUT_DIR = "/home/kerith/workspaces/smooth_simple"
DEFAULT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "prediction_timing_benchmark.csv")
SPLIT_SEED = 42
DEFAULT_START_SEED = 17
DEFAULT_MODELS_PER_DEPTH = 25
DEFAULT_DEPTHS = [1, 2, 3, 4, 5, 6]


TASKS = [
    {
        "name": "california_housing",
        "data_path": "/home/kerith/workspaces/test_output/regression/california_housing.csv",
        "model_class": cb.CatBoostRegressor,
        "base_params": {
            "loss_function": "Poisson",
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
        "interpolation_params": {
            "interpolation_enabled": True,
            "interpolation_type": "Sigmoid",
            "interpolation_min_span": {
                "AveBedrms": 0.2,
                "AveOccup": 0.0001,
                "AveRooms": 0.0001,
                "HouseAge": 0.0001,
                "MedInc": 2,
                "Population": 0.0001,
            },
            "interpolation_span_mode": {
                "AveBedrms": "Relative",
                "AveOccup": "Relative",
                "AveRooms": "Relative",
                "HouseAge": "Relative",
                "Latitude": "Absolute",
                "Longitude": "Absolute",
                "MedInc": "Relative",
                "Population": "Relative",
            },
            "interpolation_span": {
                "AveBedrms": 0.05,
                "AveOccup": 0.2,
                "AveRooms": 0.16,
                "HouseAge": 0.25,
                "Latitude": 1,
                "Longitude": 1,
                "MedInc": 0.25,
                "Population": 0.25,
            },
        },
    },
    {
        "name": "breast_cancer",
        "data_path": "/home/kerith/workspaces/test_output/classification/breast_cancer.csv",
        "model_class": cb.CatBoostClassifier,
        "base_params": {
            "loss_function": "Logloss",
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
        "interpolation_params": {
            "interpolation_enabled": True,
            "interpolation_type": "Sigmoid",
            "interpolation_min_span": {
                "area error": 0,
                "mean concave points": 0,
                "mean texture": 0,
                "perimeter error": 0,
                "worst area": 0,
                "worst concave points": 0,
                "worst concavity": 0,
                "worst perimeter": 0,
                "worst radius": 0,
                "worst texture": 0,
            },
            "interpolation_span_mode": {
                "area error": "Relative",
                "mean concave points": "Relative",
                "mean texture": "Relative",
                "perimeter error": "Relative",
                "radius error": "Absolute",
                "worst area": "Relative",
                "worst concave points": "Relative",
                "worst concavity": "Relative",
                "worst perimeter": "Relative",
                "worst radius": "Relative",
                "worst texture": "Relative",
                "mean fractal dimension": "Absolute",
                "worst symmetry": "Absolute",
                "compactness error": "Absolute",
            },
            "interpolation_span": {
                "area error": 0.25,
                "mean concave points": 0.5,
                "mean texture": 0.2,
                "perimeter error": 0.3,
                "radius error": 1,
                "worst area": 0.5,
                "worst concave points": 0.5,
                "worst concavity": 1,
                "worst perimeter": 0.5,
                "worst radius": 0.25,
                "worst texture": 0.3,
                "mean fractal dimension": 0.025,
                "worst symmetry": 0.025,
                "compactness error": 0.01,
            },
        },
    },
]


FIELDNAMES = [
    "dataset",
    "depth",
    "starting_seed",
    "interpolation_enabled",
    "prediction_duration_seconds",
    "n_rows_scored",
    "n_features",
    "tree_count",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark whole-dataset CatBoost prediction time with and without interpolation."
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


def build_params(task, depth, seed, interpolation_enabled):
    params = copy.deepcopy(task["base_params"])
    if interpolation_enabled:
        params.update(copy.deepcopy(task["interpolation_params"]))

    params.update(
        {
            "max_depth": depth,
            "n_estimators": 1000,
            "random_seed": seed,
        }
    )
    return params


def measure_prediction_duration(model, X):
    gc.collect()
    start = perf_counter()
    predictions = model.predict(X)
    duration = perf_counter() - start

    # Touch the result so timing includes materializing CatBoost's output array.
    if len(predictions) != len(X):
        raise RuntimeError(
            f"Expected {len(X)} predictions, received {len(predictions)}"
        )
    return duration


def load_task_data(task):
    df = pd.read_csv(task["data_path"])
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def run_one_model(task, X, y, depth, seed, interpolation_enabled):
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=SPLIT_SEED,
    )
    params = build_params(task, depth, seed, interpolation_enabled)
    model = task["model_class"](**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=25,
        verbose=False,
    )

    duration = measure_prediction_duration(model, X)
    return {
        "dataset": task["name"],
        "depth": depth,
        "starting_seed": seed,
        "interpolation_enabled": interpolation_enabled,
        "prediction_duration_seconds": duration,
        "n_rows_scored": len(X),
        "n_features": len(X.columns),
        "tree_count": model.tree_count_,
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
            for depth in args.depths:
                for interpolation_enabled in [False, True]:
                    for seed in model_seeds(args.start_seed, args.models_per_depth):
                        print(
                            "Training "
                            f"dataset={task['name']} "
                            f"depth={depth} "
                            f"seed={seed} "
                            f"interpolation_enabled={interpolation_enabled}",
                            flush=True,
                        )
                        row = run_one_model(
                            task,
                            X,
                            y,
                            depth,
                            seed,
                            interpolation_enabled,
                        )
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

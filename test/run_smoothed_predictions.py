import os

import catboost as cb
import pandas as pd
from sklearn.model_selection import train_test_split


OUTPUT_DIR = "/home/kerith/workspaces/smooth_simple"
SPLIT_SEED = 42
MODEL_SEEDS = [17, 18]
DEPTHS = [1, 2, 3]


TASKS = [
    {
        "name": "california_housing",
        "data_path": "/home/kerith/workspaces/test_output/regression/california_housing.csv",
        "model_class": cb.CatBoostRegressor,
        "predict_kind": "raw",
        "base_params": {
            "loss_function": "Poisson",
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
    },
    {
        "name": "breast_cancer",
        "data_path": "/home/kerith/workspaces/test_output/classification/breast_cancer.csv",
        "model_class": cb.CatBoostClassifier,
        "predict_kind": "positive_probability",
        "base_params": {
            "loss_function": "Logloss",
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
    },
]


def predict(model, X, predict_kind):
    if predict_kind == "positive_probability":
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def run_task(task):
    df = pd.read_csv(task["data_path"])
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=SPLIT_SEED,
    )

    for depth in DEPTHS:
        predictions = pd.DataFrame(
            {
                "row_id": X_val.index,
                "target": y_val.to_numpy(),
            }
        )

        for model_seed in MODEL_SEEDS:
            print(f"Training smoothed {task['name']} depth={depth} seed={model_seed}")
            params = dict(task["base_params"])
            params.update(
                {
                    "max_depth": depth,
                    "n_estimators": 1000,
                    "random_seed": model_seed,
                }
            )

            model = task["model_class"](**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=25,
                verbose=False,
            )

            predictions[f"pred_seed_{model_seed}"] = predict(
                model,
                X_val,
                task["predict_kind"],
            )

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{task['name']}_smoothed_depth_{depth}_predictions.csv",
        )
        predictions.to_csv(output_path, index=False)
        print(f"Saved {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for task in TASKS:
        run_task(task)


if __name__ == "__main__":
    main()

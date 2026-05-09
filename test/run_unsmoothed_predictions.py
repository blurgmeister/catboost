import os

import catboost as cb
import pandas as pd
from sklearn.model_selection import train_test_split


OUTPUT_DIR = "/home/kerith/workspaces/smooth_simple"
SPLIT_SEED = 42
MODEL_SEEDS = [22, 33]
DEPTHS = [1, 2, 3]


TASKS = [
    {
        "name": "california_housing",
        "data_path": "/home/kerith/workspaces/test_output/regression/california_housing.csv",
        "model_class": cb.CatBoostRegressor,
        "predict_kind": "raw",
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
    },
    {
        "name": "breast_cancer",
        "data_path": "/home/kerith/workspaces/test_output/classification/breast_cancer.csv",
        "model_class": cb.CatBoostClassifier,
        "predict_kind": "positive_probability",
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
            print(f"Training unsmoothed {task['name']} depth={depth} seed={model_seed}")
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
            f"{task['name']}_unsmoothed_depth_{depth}_predictions.csv",
        )
        predictions.to_csv(output_path, index=False)
        print(f"Saved {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for task in TASKS:
        run_task(task)


if __name__ == "__main__":
    main()

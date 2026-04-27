import os

import catboost as cb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence
from sklearn.model_selection import train_test_split


matplotlib.use("Agg")


def _feature_refs(data):
    return [data[column].to_numpy() for column in data.columns]


def _rescaled_pdp_ice(model, data, feature, random_seed):
    pd_results = partial_dependence(
        model,
        data,
        [feature],
        kind="both",
        grid_resolution=50,
    )

    grid_values = pd_results["grid_values"][0]
    ice_raw = pd_results["individual"]
    pdp_raw = pd_results["average"]

    ice_preds = ice_raw[0] if ice_raw.ndim == 3 else ice_raw
    pdp_preds = pdp_raw[0] if pdp_raw.ndim == 2 else pdp_raw

    ref_val = data[feature].median()
    idx_ref = np.argmin(np.abs(grid_values - ref_val))

    denom = ice_preds[:, idx_ref].copy()
    denom[np.abs(denom) < 1e-9] = 1e-9
    ice_rescaled = ice_preds / denom[:, None]

    pdp_denom = pdp_preds[idx_ref]
    if abs(pdp_denom) < 1e-9:
        pdp_denom = 1e-9
    pdp_rescaled = pdp_preds / pdp_denom

    rng = np.random.default_rng(random_seed)
    n_ice_plot = min(100, ice_rescaled.shape[0])
    ice_indices = rng.choice(ice_rescaled.shape[0], n_ice_plot, replace=False)

    return grid_values, ice_rescaled, pdp_rescaled, ice_indices


def _plot_feature(model, data, feature, mode_name, output_dir, random_seed):
    grid_values, ice_rescaled, pdp_rescaled, ice_indices = _rescaled_pdp_ice(
        model,
        data,
        feature,
        random_seed,
    )

    plt.figure(figsize=(10, 6))
    for idx in ice_indices:
        plt.plot(grid_values, ice_rescaled[idx], color="gray", alpha=0.1, linewidth=0.5)

    plt.plot(grid_values, pdp_rescaled, color="blue", linewidth=2, label="PDP (Rescaled)")
    plt.title(f"Depth 2 - {mode_name} span mode - Feature: {feature} (Rescaled at median)")
    plt.xlabel(feature)
    plt.ylabel("Relative Prediction Change")
    plt.legend()

    png_path = os.path.join(output_dir, f"depth_2_{mode_name.lower()}_{feature}_pdp_ice.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"    Wrote {png_path}")


def run_span_mode_test():
    random_seed = 42
    data_path = "/home/kerith/workspaces/test_output/regression/california_housing.csv"
    output_dir = "/home/kerith/workspaces/test_output/span_mode_test/"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    df["Latitude"] = (df["Latitude"] - 33.0) * 1000.0
    df["Longitude"] = (df["Longitude"] + 120.0) * 1000.0

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=random_seed,
    )

    feature_names = ["Latitude", "Longitude"]
    feature_indices = {feature: X.columns.get_loc(feature) for feature in feature_names}

    runs = [
        {
            "mode_name": "Relative",
            "interpolation_span_mode": {feature: "Relative" for feature in feature_names},
            "interpolation_span_per_float_feature": {
                feature_indices[feature]: 0.25 for feature in feature_names
            },
        },
        {
            "mode_name": "Absolute",
            "interpolation_span_mode": {feature: "Absolute" for feature in feature_names},
            "interpolation_span_per_float_feature": {
                feature_indices[feature]: 10.0 for feature in feature_names
            },
        },
    ]

    base_params = {
        "loss_function": "Poisson",
        "random_seed": random_seed,
        "interpolation_enabled": True,
        "interpolation_type": "Sigmoid",
        "max_depth": 2,
        "n_estimators": 1000,
        "task_type": "CPU",
        "monotone_constraints": {
            "AveBedrms": 1,
            "AveOccup": -1,
            "HouseAge": 1,
            "Latitude": -1,
            "Longitude": -1,
            "MedInc": 1,
            "Population": 1,
        },
    }

    for run in runs:
        print(f"\nTraining depth-2 CPU regression model with {run['mode_name']} span mode...")
        params = base_params.copy()
        params.update(
            {
                "interpolation_span_mode": run["interpolation_span_mode"],
                "interpolation_span_per_float_feature": run["interpolation_span_per_float_feature"],
            }
        )

        model = cb.CatBoostRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=25,
            verbose=False,
        )

        for feature in feature_names:
            print(f"  Generating {run['mode_name']} span mode plot for {feature}...")
            _plot_feature(model, X, feature, run["mode_name"], output_dir, random_seed)


if __name__ == "__main__":
    run_span_mode_test()

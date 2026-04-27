import json
import tempfile
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split


DATA_PATH = Path("/workspaces/test_output/regression/california_housing.csv")
TARGET_COLUMN = "target"
FEATURE_NAME = "AveBedrms"
SPAN = 0.005
SPAN_MODE = "Absolute"
GRID_SIZE = 51
RANDOM_SEED = 42


def load_split_borders(model, feature_name):
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        model.save_model(tmp.name, format="json")
        model_json = json.loads(Path(tmp.name).read_text())

    float_feature_index = None
    for feature in model_json["features_info"]["float_features"]:
        if feature["feature_id"] == feature_name:
            float_feature_index = feature["feature_index"]
            break
    if float_feature_index is None:
        raise RuntimeError(f"Feature {feature_name!r} not found in saved model")

    borders = []
    for tree in model_json["oblivious_trees"]:
        for split in tree["splits"]:
            if split["split_type"] != "FloatFeature":
                continue
            if split["float_feature_index"] == float_feature_index:
                borders.append(float(split["border"]))
    return float_feature_index, np.array(borders, dtype=float)


def pdp_raw(model, X, feature_name, grid, split_borders, span, span_mode):
    values = []
    active_counts = []
    if span_mode == "Absolute":
        spans = np.full_like(split_borders, span, dtype=float)
    else:
        spans = np.maximum(0.0, np.abs(split_borders) * span)
    for value in grid:
        x_eval = X.copy()
        x_eval[feature_name] = value
        preds = np.asarray(model.predict(x_eval, prediction_type="RawFormulaVal"))
        values.append(preds.mean())
        if split_borders.size:
            active_counts.append(int(np.count_nonzero(np.abs(split_borders - value) < spans)))
        else:
            active_counts.append(0)
    return np.array(values), np.array(active_counts, dtype=int)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", default=FEATURE_NAME)
    parser.add_argument("--span", type=float, default=SPAN)
    parser.add_argument("--span-mode", choices=["Absolute", "Relative"], default=SPAN_MODE)
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    print("loaded_data", X.shape, flush=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED
    )
    print("split_data", X_train.shape, X_val.shape, flush=True)

    base_params = dict(
        loss_function="Poisson",
        random_seed=RANDOM_SEED,
        monotone_constraints={
            "AveBedrms": 1,
            "AveOccup": -1,
            "HouseAge": 1,
            "Latitude": -1,
            "Longitude": -1,
            "MedInc": 1,
            "Population": 1,
        },
        max_depth=1,
        n_estimators=1000,
        verbose=False,
    )

    hard_model = CatBoostRegressor(**base_params)
    print("fitting_hard_model", flush=True)
    hard_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=25,
        verbose=False,
    )
    print("fitted_hard_model", hard_model.tree_count_, flush=True)

    interpolated_model = CatBoostRegressor(
        **base_params,
        interpolation_enabled=True,
        interpolation_type="Linear",
        interpolation_span_mode={args.feature: args.span_mode},
        interpolation_span={X.columns.get_loc(args.feature): args.span},
        interpolation_min_span=0.0,
    )
    print("fitting_interpolated_model", flush=True)
    interpolated_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=25,
        verbose=False,
    )
    print("fitted_interpolated_model", interpolated_model.tree_count_, flush=True)

    feature_idx, split_borders = load_split_borders(interpolated_model, args.feature)
    sorted_borders = np.sort(split_borders)
    gaps = np.diff(sorted_borders) if len(sorted_borders) > 1 else np.array([], dtype=float)

    feature_values = X[args.feature].to_numpy()
    grid = np.linspace(feature_values.min(), feature_values.max(), GRID_SIZE)
    print("starting_pdp_sweeps", GRID_SIZE, flush=True)

    hard_pdp, _ = pdp_raw(hard_model, X, args.feature, grid, split_borders, args.span, args.span_mode)
    interp_pdp, active_counts = pdp_raw(interpolated_model, X, args.feature, grid, split_borders, args.span, args.span_mode)
    print("finished_pdp_sweeps", flush=True)

    outside_all_windows = np.ones_like(grid, dtype=bool)
    if args.span_mode == "Absolute":
        spans = np.full_like(split_borders, args.span, dtype=float)
    else:
        spans = np.maximum(0.0, np.abs(split_borders) * args.span)
    for border, local_span in zip(split_borders, spans):
        outside_all_windows &= np.abs(grid - border) >= local_span

    print("dataset_shape", X.shape)
    print("feature_name", args.feature)
    print("feature_index", feature_idx)
    print("span", args.span)
    print("span_mode", args.span_mode)
    print("feature_range", float(feature_values.min()), float(feature_values.max()))
    print("tree_count", interpolated_model.tree_count_)
    print("split_count_on_feature", int(len(split_borders)))
    print("first_20_sorted_borders", sorted_borders[:20].tolist())
    print("last_20_sorted_borders", sorted_borders[-20:].tolist())
    if gaps.size:
        print("min_gap", float(gaps.min()))
        print("median_gap", float(np.median(gaps)))
        if args.span_mode == "Absolute":
            print("count_gap_le_2span", int(np.count_nonzero(gaps <= 2.0 * args.span)))
            print("count_gap_le_4span", int(np.count_nonzero(gaps <= 4.0 * args.span)))
        else:
            print("count_gap_le_2span", "not_applicable_for_relative_span")
            print("count_gap_le_4span", "not_applicable_for_relative_span")
    else:
        print("min_gap", None)
        print("median_gap", None)
        print("count_gap_le_2span", 0)
        print("count_gap_le_4span", 0)

    print("grid_points", int(grid.size))
    print("grid_points_outside_all_windows", int(np.count_nonzero(outside_all_windows)))
    print("grid_fraction_outside_all_windows", float(outside_all_windows.mean()))
    print("max_active_trees_at_point", int(active_counts.max()) if active_counts.size else 0)
    print("mean_active_trees_at_point", float(active_counts.mean()) if active_counts.size else 0.0)

    print("hard_raw_pdp_min_max", float(hard_pdp.min()), float(hard_pdp.max()))
    print("interp_raw_pdp_min_max", float(interp_pdp.min()), float(interp_pdp.max()))
    print("hard_raw_pdp_range", float(hard_pdp.max() - hard_pdp.min()))
    print("interp_raw_pdp_range", float(interp_pdp.max() - interp_pdp.min()))

    if np.any(outside_all_windows):
        hard_plateau = hard_pdp[outside_all_windows]
        interp_plateau = interp_pdp[outside_all_windows]
        print("hard_plateau_min_max", float(hard_plateau.min()), float(hard_plateau.max()))
        print("interp_plateau_min_max", float(interp_plateau.min()), float(interp_plateau.max()))
        print("plateau_max_abs_diff", float(np.max(np.abs(hard_plateau - interp_plateau))))

    largest_diffs = np.argsort(np.abs(interp_pdp - hard_pdp))[-10:]
    largest_diffs = largest_diffs[np.argsort(grid[largest_diffs])]
    print("largest_diff_points")
    for idx in largest_diffs:
        print(
            float(grid[idx]),
            float(hard_pdp[idx]),
            float(interp_pdp[idx]),
            int(active_counts[idx]),
        )

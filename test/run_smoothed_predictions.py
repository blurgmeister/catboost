import os
from pathlib import Path

import catboost as cb
import pandas as pd

from openml_prediction_utils import (
    DEPTHS,
    add_prediction_columns,
    build_base_params,
    load_prepared_data,
    load_tasks,
    safe_name,
)


OUTPUT_DIR = "/home/kerith/workspaces/smooth_multi/predictions"
AVG_SPLIT_GAPS_PATH = Path("/home/kerith/workspaces/smooth_multi/avg_split_gaps.csv")
MODEL_SEEDS = [22, 33]
DEFAULT_ABSOLUTE_SPAN = 1.0
INTERPOLATION_PARAMS = {
    "interpolation_enabled": True,
    "interpolation_type": "Sigmoid",
}


def load_absolute_spans(path=AVG_SPLIT_GAPS_PATH):
    split_gaps = pd.read_csv(path)
    split_gaps["depth"] = split_gaps["depth"].astype(int)
    split_gaps["span"] = pd.to_numeric(split_gaps["selected"], errors="coerce")
    if split_gaps["span"].isna().all():
        split_gaps["span"] = pd.to_numeric(split_gaps["avg_dist"], errors="coerce")

    spans = {}
    for (dataset_name, depth), group in split_gaps.groupby(["dataset", "depth"]):
        feature_spans = {
            str(row["feature"]): float(row["span"])
            for _, row in group.iterrows()
            if pd.notna(row["span"]) and float(row["span"]) > 0
        }
        if feature_spans:
            spans[(str(dataset_name), int(depth))] = feature_spans
    return spans


def interpolation_params_for(task_name, depth, spans_by_dataset_depth):
    feature_spans = spans_by_dataset_depth.get((task_name, depth), {})
    if not feature_spans and depth > 2:
        feature_spans = spans_by_dataset_depth.get((task_name, 2), {})
    if not feature_spans:
        feature_spans = {"__default__": DEFAULT_ABSOLUTE_SPAN}
    params = dict(INTERPOLATION_PARAMS)
    params["interpolation_span"] = feature_spans
    params["interpolation_span_mode"] = {
        feature: "Absolute"
        for feature in feature_spans
    }
    return params


def map_spans_to_model_features(feature_spans, model_features):
    if feature_spans == {"__default__": DEFAULT_ABSOLUTE_SPAN}:
        return feature_spans

    exact_features = set(model_features)
    sanitized_to_feature = {}
    for feature in model_features:
        sanitized_to_feature.setdefault(safe_name(feature), feature)

    mapped = {}
    for feature, span in feature_spans.items():
        if feature in exact_features:
            mapped[feature] = span
        elif feature in sanitized_to_feature:
            mapped[sanitized_to_feature[feature]] = span
    return mapped


def run_task(task, spans_by_dataset_depth):
    prepared = load_prepared_data(task)
    model_class, base_params = build_base_params(task, prepared)

    for depth in DEPTHS:
        predictions = pd.DataFrame(
            {
                "row_id": prepared.X_val.index,
                "target": prepared.y_val.to_numpy(),
            }
        )

        for model_seed in MODEL_SEEDS:
            print(f"Training smoothed {task.name} depth={depth} seed={model_seed}")
            params = dict(base_params)
            params.update(interpolation_params_for(task.name, depth, spans_by_dataset_depth))
            params["interpolation_span"] = map_spans_to_model_features(
                params["interpolation_span"],
                prepared.X_train.columns,
            )
            params["interpolation_span_mode"] = {
                feature: "Absolute"
                for feature in params["interpolation_span"]
            }
            params.update(
                {
                    "max_depth": depth,
                    "n_estimators": 1000,
                    "random_seed": model_seed,
                }
            )

            model = model_class(**params)
            train_pool = cb.Pool(
                prepared.X_train,
                prepared.y_train,
                cat_features=prepared.cat_features,
                weight=prepared.weight_train,
            )
            eval_pool = cb.Pool(
                prepared.X_val,
                prepared.y_val,
                cat_features=prepared.cat_features,
                weight=prepared.weight_val,
            )
            model.fit(
                train_pool,
                eval_set=eval_pool,
                early_stopping_rounds=25,
                verbose=False,
            )

            add_prediction_columns(predictions, model, prepared.X_val, model_seed, task)

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{safe_name(task.name)}_smoothed_depth_{depth}_predictions.csv",
        )
        predictions.to_csv(output_path, index=False)
        print(f"Saved {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    spans_by_dataset_depth = load_absolute_spans()
    for task in load_tasks():
        run_task(task, spans_by_dataset_depth)


if __name__ == "__main__":
    main()

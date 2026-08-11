import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import catboost as cb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from openml_prediction_utils import (
    DEPTHS,
    FRENCH_MOTOR_CLAIMS_SUITE,
    build_base_params,
    load_prepared_data,
    load_tasks,
    safe_name,
)
from run_smoothed_predictions import (
    interpolation_params_for,
    load_absolute_spans,
    map_spans_to_model_features,
)


OUTPUT_DIR = Path("/home/kerith/workspaces/smooth_multi/decile")
DEFAULT_ONLY_SUITE = "all"
DEFAULT_SEED = 22
DEFAULT_ESTIMATORS = 1000


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit CatBoost with and without interpolation and create decile plots "
            "sorted by the unsmoothed model predictions."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where decile charts and chart data will be saved.",
    )
    parser.add_argument(
        "--only-suite",
        default=DEFAULT_ONLY_SUITE,
        help="Only process datasets from this suite. Defaults to all supported suites.",
    )
    parser.add_argument(
        "--only-dataset-name",
        default=None,
        help="Only process datasets whose name contains this case-insensitive text.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--depths", type=int, nargs="+", default=DEPTHS)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_ESTIMATORS)
    return parser.parse_args()


def selected_tasks(args):
    suites = None if args.only_suite.lower() == "all" else {args.only_suite}
    tasks = load_tasks() if suites is None else load_tasks(suites=suites)
    if args.only_dataset_name:
        needle = args.only_dataset_name.lower()
        tasks = [task for task in tasks if needle in task.name.lower()]
    return tasks


def build_params(task, prepared, depth, seed, n_estimators, spans_by_dataset_depth, smoothed):
    _, base_params = build_base_params(task, prepared)
    params = dict(base_params)
    if smoothed:
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
            "n_estimators": n_estimators,
            "random_seed": seed,
        }
    )
    return params


def fit_model(task, prepared, depth, seed, n_estimators, spans_by_dataset_depth, smoothed):
    model_class, _ = build_base_params(task, prepared)
    params = build_params(
        task,
        prepared,
        depth,
        seed,
        n_estimators,
        spans_by_dataset_depth,
        smoothed,
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
    return model


def model_outputs(model, prepared, task):
    if task.task_type == "regression":
        return [
            {
                "label": "",
                "prediction": np.asarray(model.predict(prepared.X_val), dtype=float),
                "response": prepared.y_val.to_numpy(dtype=float),
            }
        ]

    probabilities = np.asarray(model.predict_proba(prepared.X_val), dtype=float)
    class_labels = [str(value) for value in model.classes_]
    y_as_string = prepared.y_val.astype(str).to_numpy()
    if probabilities.shape[1] == 2:
        positive_index = 1
        positive_label = class_labels[positive_index]
        return [
            {
                "label": positive_label,
                "prediction": probabilities[:, positive_index],
                "response": (y_as_string == positive_label).astype(float),
            }
        ]

    outputs = []
    for class_index, class_label in enumerate(class_labels):
        outputs.append(
            {
                "label": class_label,
                "prediction": probabilities[:, class_index],
                "response": (y_as_string == class_label).astype(float),
            }
        )
    return outputs


def weighted_deciles(sort_values, weights):
    order = np.argsort(sort_values, kind="mergesort")
    deciles = np.empty(len(sort_values), dtype=int)
    if weights is None:
        buckets = np.minimum((np.arange(len(order)) * 10) // len(order), 9)
        deciles[order] = buckets + 1
        return deciles

    sorted_weights = np.asarray(weights, dtype=float)[order]
    sorted_weights = np.nan_to_num(sorted_weights, nan=0.0, posinf=0.0, neginf=0.0)
    total_weight = sorted_weights.sum()
    if total_weight <= 0:
        buckets = np.minimum((np.arange(len(order)) * 10) // len(order), 9)
    else:
        cumulative_midpoint = np.cumsum(sorted_weights) - (0.5 * sorted_weights)
        buckets = np.floor(cumulative_midpoint / total_weight * 10).astype(int)
        buckets = np.clip(buckets, 0, 9)
    deciles[order] = buckets + 1
    return deciles


def average(values, weights):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan
    if weights is None:
        return float(np.mean(values))
    weights = np.nan_to_num(np.asarray(weights, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if weights.sum() <= 0:
        return float(np.mean(values))
    return float(np.average(values, weights=weights))


def decile_rows(dataset, depth, output_label, response, unsmoothed, smoothed, weights):
    deciles = weighted_deciles(unsmoothed, weights)
    rows = []
    for decile in range(1, 11):
        mask = deciles == decile
        decile_weights = None if weights is None else np.asarray(weights, dtype=float)[mask]
        rows.append(
            {
                "dataset": dataset,
                "depth": depth,
                "output": output_label,
                "decile": decile,
                "row_count": int(mask.sum()),
                "weight_sum": np.nan if weights is None else float(np.sum(decile_weights)),
                "response": average(response[mask], decile_weights),
                "unsmoothed_prediction": average(unsmoothed[mask], decile_weights),
                "smoothed_prediction": average(smoothed[mask], decile_weights),
            }
        )
    return rows


def plot_deciles(task, depth, output_label, rows, output_dir):
    chart_data = pd.DataFrame(rows)
    title_suffix = "" if output_label == "" else f" | label {output_label}"
    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(chart_data["decile"], chart_data["response"], marker="o", label="Response")
    axis.plot(
        chart_data["decile"],
        chart_data["unsmoothed_prediction"],
        marker="o",
        label="Unsmoothed prediction",
    )
    axis.plot(
        chart_data["decile"],
        chart_data["smoothed_prediction"],
        marker="o",
        label="Smoothed prediction",
    )
    axis.set_title(f"{task.name}{title_suffix} | depth {depth}")
    axis.set_xlabel("Prediction decile sorted by unsmoothed model")
    axis.set_ylabel("Weighted average" if task.suite == FRENCH_MOTOR_CLAIMS_SUITE else "Average")
    axis.set_xticks(range(1, 11))
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()

    class_suffix = "" if output_label == "" else f"_label_{safe_name(output_label)}"
    output_path = output_dir / f"{safe_name(task.name)}_depth_{depth}{class_suffix}_decile.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def run_task(task, args, spans_by_dataset_depth):
    prepared = load_prepared_data(task)
    weights = None if prepared.weight_val is None else prepared.weight_val.to_numpy(dtype=float)
    all_rows = []

    if task.suite == FRENCH_MOTOR_CLAIMS_SUITE:
        print(f"Using exposure weights for fitting and weighted decile grouping: {task.name}")

    for depth in args.depths:
        print(f"Training unsmoothed {task.name} depth={depth} seed={args.seed}")
        unsmoothed_model = fit_model(
            task,
            prepared,
            depth,
            args.seed,
            args.n_estimators,
            spans_by_dataset_depth,
            smoothed=False,
        )
        print(f"Training smoothed {task.name} depth={depth} seed={args.seed}")
        smoothed_model = fit_model(
            task,
            prepared,
            depth,
            args.seed,
            args.n_estimators,
            spans_by_dataset_depth,
            smoothed=True,
        )

        unsmoothed_outputs = model_outputs(unsmoothed_model, prepared, task)
        smoothed_outputs = {
            output["label"]: output
            for output in model_outputs(smoothed_model, prepared, task)
        }
        for unsmoothed_output in unsmoothed_outputs:
            output_label = unsmoothed_output["label"]
            smoothed_output = smoothed_outputs[output_label]
            rows = decile_rows(
                task.name,
                depth,
                output_label,
                unsmoothed_output["response"],
                unsmoothed_output["prediction"],
                smoothed_output["prediction"],
                weights,
            )
            all_rows.extend(rows)
            plot_deciles(task, depth, output_label, rows, args.output_dir)

    return all_rows


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spans_by_dataset_depth = load_absolute_spans()
    tasks = selected_tasks(args)
    if not tasks:
        print("No datasets matched the requested filters.")
        return

    all_rows = []
    for task in tasks:
        all_rows.extend(run_task(task, args, spans_by_dataset_depth))

    chart_data_path = args.output_dir / "decile_chart_data.csv"
    pd.DataFrame(all_rows).to_csv(chart_data_path, index=False)
    print(f"Saved {chart_data_path}")


if __name__ == "__main__":
    main()

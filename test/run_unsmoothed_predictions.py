import argparse
import os

import catboost as cb
import pandas as pd

from openml_prediction_utils import (
    DEPTHS,
    build_base_params,
    add_prediction_columns,
    load_prepared_data,
    load_tasks,
    safe_name,
)


OUTPUT_DIR = "/home/kerith/workspaces/smooth_multi/predictions"
MODEL_SEEDS = [22, 33]
DEFAULT_ONLY_SUITE = "all"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train unsmoothed CatBoost models and save validation predictions."
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
    return parser.parse_args()


def selected_tasks(args):
    suites = None if args.only_suite.lower() == "all" else {args.only_suite}
    tasks = load_tasks() if suites is None else load_tasks(suites=suites)
    if args.only_dataset_name:
        needle = args.only_dataset_name.lower()
        tasks = [task for task in tasks if needle in task.name.lower()]
    return tasks


def run_task(task):
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
            print(f"Training unsmoothed {task.name} depth={depth} seed={model_seed}")
            params = dict(base_params)
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
            f"{safe_name(task.name)}_unsmoothed_depth_{depth}_predictions.csv",
        )
        predictions.to_csv(output_path, index=False)
        print(f"Saved {output_path}")


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tasks = selected_tasks(args)
    if not tasks:
        print("No datasets matched the requested filters.")
        return
    for task in tasks:
        run_task(task)


if __name__ == "__main__":
    main()

"""Train one unsmoothed model per dataset/depth and inspect its symmetric trees.

Trees and depths are numbered from 1 (root depth = 1). Reuse means an input
feature occurs at two or more distinct depths of a tree, including input
features underlying categorical CTR combinations. Each such tree counts once.
Constant trees count in the denominator and have one blank-depth detail row.
CBM and JSON models, run metadata, CSVs, and the plot are saved together.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import catboost as cb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openml_prediction_utils import (
    DEPTHS, SPLIT_SEED, build_base_params, load_prepared_data, load_tasks, safe_name,
)


OUTPUT_DIR = Path("/home/kerith/workspaces/smooth_multi/example_models")
DETAIL_FIELDS = [
    "suite", "dataset", "max_depth", "random_seed", "split_seed", "model_file",
    "tree", "depth", "feature", "feature_index", "split_type", "split_details",
]
SUMMARY_FIELDS = [
    "suite", "dataset", "max_depth", "random_seed", "split_seed", "model_file",
    "trees_with_repeated_feature", "total_trees", "proportion",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-suite", default="all")
    parser.add_argument("--only-dataset-name", default=None)
    parser.add_argument("--seed", type=int, default=0,
                        help="Starting model seed; incremented for each dataset/depth run.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--thread-count", type=int, default=-1)
    return parser.parse_args()


def split_features(model_json):
    """Map binary split indexes to original input feature indexes and names."""
    info = model_json["features_info"]
    floats = {f["feature_index"]: f for f in info.get("float_features", [])}
    cats = {f["feature_index"]: f for f in info.get("categorical_features", [])}
    result = {}
    index = 0
    for feature in info.get("float_features", []):
        for _ in feature.get("borders", []):
            result[index] = [feature]
            index += 1
    for feature in info.get("categorical_features", []):
        for _ in feature.get("values", []):
            result[index] = [feature]
            index += 1
    for ctr in info.get("ctrs", []):
        features = {}
        for element in ctr["elements"]:
            if "cat_feature_index" in element:
                feature = cats[element["cat_feature_index"]]
            else:
                feature = floats[element["float_feature_index"]]
            features[feature["flat_feature_index"]] = feature
        for _ in ctr["borders"]:
            result[index] = list(features.values())
            index += 1
    return result


def examine_model(model_json, run):
    if "oblivious_trees" not in model_json:
        raise ValueError("Expected a symmetric (oblivious) tree model.")
    features_by_split = split_features(model_json)
    rows = []
    repeated = 0
    trees = model_json["oblivious_trees"]
    for tree_number, tree in enumerate(trees, start=1):
        counts = Counter()
        # CatBoost stores leaf-bit order; its tree visualizer reverses this
        # sequence to display the root first.
        for depth, split in enumerate(reversed(tree.get("splits") or []), start=1):
            features = features_by_split[split["split_index"]]
            for feature in features:
                feature_index = feature["flat_feature_index"]
                counts[feature_index] += 1
                rows.append({
                    **run, "tree": tree_number, "depth": depth,
                    "feature": feature.get("feature_id") or str(feature_index),
                    "feature_index": feature_index, "split_type": split["split_type"],
                    "split_details": json.dumps(split, sort_keys=True),
                })
        if not tree.get("splits"):
            rows.append({**run, "tree": tree_number})
        repeated += any(count > 1 for count in counts.values())
    summary = {
        **run, "trees_with_repeated_feature": repeated, "total_trees": len(trees),
        "proportion": repeated / len(trees) if trees else 0.0,
    }
    return rows, summary


def plot_summary(summaries, output_dir):
    fig, ax = plt.subplots(figsize=(12, 7))
    groups = sorted({(row["suite"], row["dataset"]) for row in summaries})
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*", "h", "p"]
    colors = plt.get_cmap("tab20")
    for index, (suite, dataset) in enumerate(groups):
        points = [r for r in summaries if (r["suite"], r["dataset"]) == (suite, dataset)
                  and r["proportion"] > 0]
        if points:
            ax.scatter([r["max_depth"] for r in points],
                       [r["proportion"] for r in points],
                       color=colors(index % 20), marker=markers[index % len(markers)],
                       label=f"{dataset} ({suite})", alpha=0.8, s=45)
    ax.set(xlabel="Model max_depth", ylabel="Proportion of trees reusing a feature",
           title="Feature reuse within trees (nonzero proportions only)",
           xticks=DEPTHS, ylim=(0, 1.03))
    ax.grid(alpha=0.25)
    if ax.collections:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    else:
        ax.text(0.5, 0.5, "No nonzero proportions", transform=ax.transAxes, ha="center")
    fig.savefig(output_dir / "tree_feature_reuse.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if args.iterations < 1 or args.seed < 0:
        raise ValueError("iterations must be positive and seed must be nonnegative")
    suites = None if args.only_suite.lower() == "all" else {args.only_suite}
    tasks = load_tasks() if suites is None else load_tasks(suites=suites)
    if args.only_dataset_name:
        tasks = [t for t in tasks if args.only_dataset_name.lower() in t.name.lower()]
    if not tasks:
        print("No datasets matched the requested filters.")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    with (args.output_dir / "tree_details.csv").open("w", newline="") as detail_file, \
         (args.output_dir / "tree_summary.csv").open("w", newline="") as summary_file:
        detail_writer = csv.DictWriter(detail_file, fieldnames=DETAIL_FIELDS)
        summary_writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        detail_writer.writeheader()
        summary_writer.writeheader()
        run_index = 0
        for task in tasks:
            prepared = load_prepared_data(task)
            model_class, base_params = build_base_params(task, prepared)
            train_pool = cb.Pool(prepared.X_train, prepared.y_train,
                                 cat_features=prepared.cat_features, weight=prepared.weight_train)
            eval_pool = cb.Pool(prepared.X_val, prepared.y_val,
                                cat_features=prepared.cat_features, weight=prepared.weight_val)
            for depth in DEPTHS:
                seed = args.seed + run_index
                run_index += 1
                stem = f"{safe_name(task.suite)}__{safe_name(task.name)}_depth_{depth}_seed_{seed}"
                params = {**base_params, "max_depth": depth, "n_estimators": args.iterations,
                          "random_seed": seed, "thread_count": args.thread_count}
                print(f"Training {task.name} max_depth={depth} seed={seed}", flush=True)
                model = model_class(**params)
                model.fit(train_pool, eval_set=eval_pool, early_stopping_rounds=25, verbose=False)
                model_path = args.output_dir / f"{stem}.cbm"
                json_path = args.output_dir / f"{stem}.json"
                model.save_model(str(model_path))
                model.save_model(str(json_path), format="json", pool=train_pool)
                run = dict(suite=task.suite, dataset=task.name, max_depth=depth,
                           random_seed=seed, split_seed=SPLIT_SEED, model_file=model_path.name)
                with json_path.open() as source:
                    rows, summary = examine_model(json.load(source), run)
                if summary["total_trees"] != model.tree_count_:
                    raise ValueError("Exported tree count differs from fitted model")
                metadata = {**run, "dataset_path": str(task.csv_path),
                            "catboost_version": cb.__version__, "training_params": params,
                            "fitted_params": model.get_all_params(), "validation_fraction": 0.3,
                            "early_stopping_rounds": 25}
                with (args.output_dir / f"{stem}_run.json").open("w") as target:
                    json.dump(metadata, target, indent=2)
                detail_writer.writerows(rows)
                summary_writer.writerow(summary)
                detail_file.flush()
                summary_file.flush()
                summaries.append(summary)
                plot_summary(summaries, args.output_dir)
                print(f"Saved {model_path}: {summary['trees_with_repeated_feature']}/"
                      f"{summary['total_trees']} trees reuse a feature", flush=True)
    print(f"Saved tree_details.csv, tree_summary.csv and tree_feature_reuse.png in {args.output_dir}")


if __name__ == "__main__":
    main()

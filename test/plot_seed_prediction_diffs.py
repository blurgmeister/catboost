import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = "/home/kerith/workspaces/smooth_simple"
MODEL_SEEDS = [17, 18, 22, 33]
DEPTHS = [1, 2, 3]
TASKS = ["california_housing", "breast_cancer"]
MODEL_KINDS = ["unsmoothed", "smoothed"]
EPSILON = 1e-12
RATIO_CHANGE_COLUMN = "prediction_ratio_change"


def add_prediction_ratio_change(predictions, use_log_predictions=False):
    if predictions.shape[1] < 4:
        raise ValueError(
            "Prediction files must have at least 4 columns: row id, target, "
            "first model prediction, second model prediction."
        )

    first_prediction = predictions.iloc[:, 2].to_numpy()
    second_prediction = predictions.iloc[:, 3].to_numpy()
    if use_log_predictions:
        first_prediction = np.log(first_prediction)
        second_prediction = np.log(second_prediction)

    denominator = np.where(np.abs(first_prediction) < EPSILON, np.nan, first_prediction)
    predictions[RATIO_CHANGE_COLUMN] = second_prediction / denominator - 1.0
    return predictions


def ratio_change_values(predictions, use_log_predictions=False):
    predictions = add_prediction_ratio_change(predictions, use_log_predictions)
    return predictions[RATIO_CHANGE_COLUMN].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()


def load_differences(task_name, model_kind, depth):
    path = os.path.join(
        OUTPUT_DIR,
        f"{task_name}_{model_kind}_depth_{depth}_predictions.csv",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing prediction file: {path}. Run both prediction scripts first."
        )

    predictions = pd.read_csv(path)
    return ratio_change_values(
        predictions,
        use_log_predictions=(task_name == "breast_cancer"),
    )


def add_summary(summary_rows, task_name, model_kind, depth, diffs):
    summary_rows.append(
        {
            "dataset": task_name,
            "model_kind": model_kind,
            "depth": depth,
            "count": len(diffs),
            "mean_ratio_change": np.mean(diffs),
            "median_ratio_change": np.median(diffs),
            "p10_ratio_change": np.percentile(diffs, 10),
            "p90_ratio_change": np.percentile(diffs, 90),
            "p95_ratio_change": np.percentile(diffs, 95),
            "p99_abs_ratio_change": np.percentile(np.abs(diffs), 99),
            "max_abs_ratio_change": np.max(np.abs(diffs)),
        }
    )


def plot_task(task_name, summary_rows):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(DEPTHS),
        figsize=(17, 5),
        sharey=True,
    )
    fig.suptitle(f"{task_name}: row-wise prediction ratio changes by model seed")

    for axis, depth in zip(axes, DEPTHS):
        depth_diffs = {}
        for model_kind in MODEL_KINDS:
            diffs = load_differences(task_name, model_kind, depth)
            depth_diffs[model_kind] = diffs
            add_summary(summary_rows, task_name, model_kind, depth, diffs)

        combined = np.concatenate(list(depth_diffs.values()))
        lower = np.percentile(combined, 1)
        upper = np.percentile(combined, 99)
        if lower == upper:
            spread = max(abs(lower), 1.0)
            lower -= spread
            upper += spread
        bins = np.linspace(lower, upper, 50)

        for model_kind in MODEL_KINDS:
            axis.hist(
                np.clip(depth_diffs[model_kind], lower, upper),
                bins=bins,
                alpha=0.55,
                density=True,
                label=model_kind,
            )

        axis.set_title(f"Depth {depth}")
        axis.set_xlabel("Prediction ratio change")
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Density")
    axes[-1].legend()
    fig.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{task_name}_seed_prediction_ratio_change_hist.png",
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    summary_rows = []
    for task_name in TASKS:
        plot_task(task_name, summary_rows)

    summary_path = os.path.join(OUTPUT_DIR, "seed_prediction_ratio_change_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()

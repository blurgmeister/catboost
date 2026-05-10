import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


CATBOOST_INPUT_PATH = "/home/kerith/workspaces/smooth_simple/prediction_timing_benchmark.csv"
OTHER_INPUT_PATH = (
    "/home/kerith/workspaces/smooth_simple/prediction_timing_benchmark_other.csv"
)
OUTPUT_DIR = "/home/kerith/workspaces/smooth_simple"


DATASET_TITLES = {
    "california_housing": "California Housing",
    "breast_cancer": "Breast Cancer",
}

SERIES_ORDER = [
    "catboost_unsmoothed",
    "catboost_smoothed",
    "xgboost",
    "lightgbm",
]

SERIES_LABELS = {
    "catboost_unsmoothed": "CatBoost, interpolation disabled",
    "catboost_smoothed": "CatBoost, interpolation enabled",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

SERIES_COLORS = {
    "catboost_unsmoothed": "#4C78A8",
    "catboost_smoothed": "#F58518",
    "xgboost": "#54A24B",
    "lightgbm": "#B279A2",
}


def load_catboost_timings():
    df = pd.read_csv(CATBOOST_INPUT_PATH)
    df["series"] = df["interpolation_enabled"].map(
        {
            False: "catboost_unsmoothed",
            True: "catboost_smoothed",
        }
    )
    return df[["dataset", "depth", "series", "prediction_duration_seconds"]]


def load_other_timings():
    df = pd.read_csv(OTHER_INPUT_PATH)
    df["series"] = df["model_type"]
    return df[["dataset", "depth", "series", "prediction_duration_seconds"]]


def load_summary():
    df = pd.concat(
        [load_catboost_timings(), load_other_timings()],
        ignore_index=True,
    )
    df["prediction_duration_ms"] = df["prediction_duration_seconds"] * 1000

    return (
        df.groupby(["dataset", "depth", "series"], as_index=False)
        .agg(
            mean_prediction_duration_ms=("prediction_duration_ms", "mean"),
            sem_prediction_duration_ms=("prediction_duration_ms", "sem"),
            n_models=("prediction_duration_ms", "size"),
        )
        .sort_values(["dataset", "series", "depth"])
    )


def plot_dataset(summary, dataset):
    dataset_summary = summary[summary["dataset"] == dataset]
    if dataset_summary.empty:
        raise ValueError(f"No benchmark rows found for dataset={dataset}")

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for series_name in SERIES_ORDER:
        series = dataset_summary[dataset_summary["series"] == series_name].sort_values(
            "depth"
        )
        if series.empty:
            continue

        ax.errorbar(
            series["depth"],
            series["mean_prediction_duration_ms"],
            yerr=series["sem_prediction_duration_ms"],
            marker="o",
            linewidth=2,
            capsize=4,
            label=SERIES_LABELS[series_name],
            color=SERIES_COLORS[series_name],
        )

    ax.set_title(f"{DATASET_TITLES.get(dataset, dataset)} Prediction Timing")
    ax.set_xlabel("Model depth")
    ax.set_ylabel("Prediction duration (ms)")
    ax.set_xticks(sorted(dataset_summary["depth"].unique()))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    output_path = os.path.join(OUTPUT_DIR, f"{dataset}_prediction_timing_by_depth.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = load_summary()

    output_paths = []
    for dataset in sorted(summary["dataset"].unique()):
        output_paths.append(plot_dataset(summary, dataset))

    summary_path = os.path.join(OUTPUT_DIR, "prediction_timing_benchmark_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"Saved summary: {summary_path}")
    for output_path in output_paths:
        print(f"Saved chart: {output_path}")


if __name__ == "__main__":
    main()

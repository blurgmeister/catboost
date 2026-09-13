#!/usr/bin/env python3
"""Create depth-based scatter charts for prediction-difference ratio columns."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


INPUT_CSV = Path(
    "/home/kerith/workspaces/smooth_multi/predictions/"
    "prediction_difference_width_reductions.csv"
)
OUTPUT_DIR = Path("/home/kerith/workspaces/smooth_multi/predictions")
SUMMARY_FILENAME = "prediction_difference_ratio_summary.csv"


def create_ratio_charts(input_csv: Path = INPUT_CSV, output_dir: Path = OUTPUT_DIR) -> list[Path]:
    """Create one scatter chart per ``_ratio_`` column and return output paths."""
    data = pd.read_csv(input_csv)

    if "depth" not in data.columns:
        raise ValueError(f"Required column 'depth' was not found in {input_csv}")

    ratio_columns = [column for column in data.columns if "_ratio_" in column]
    if not ratio_columns:
        raise ValueError(f"No columns containing '_ratio_' were found in {input_csv}")

    data["depth"] = pd.to_numeric(data["depth"], errors="coerce")
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_charts: list[Path] = []
    summary_frames: list[pd.DataFrame] = []

    for column in ratio_columns:
        plot_data = data[["depth", column]].copy()
        plot_data[column] = pd.to_numeric(plot_data[column], errors="coerce")
        plot_data = plot_data.dropna()

        if plot_data.empty:
            print(f"Skipping {column}: it contains no numeric values.")
            continue

        by_depth = plot_data.groupby("depth", as_index=False)[column].agg(
            mean="mean", median="median"
        )
        summary = plot_data.groupby("depth")[column].agg(
            mean="mean",
            median="median",
            percentile_2_5=lambda values: values.quantile(0.025),
            percentile_5=lambda values: values.quantile(0.05),
            percentile_25=lambda values: values.quantile(0.25),
            percentile_75=lambda values: values.quantile(0.75),
            percentile_95=lambda values: values.quantile(0.95),
            percentile_97_5=lambda values: values.quantile(0.975),
        ).reset_index()
        summary.insert(0, "ratio_metric", column)
        summary_frames.append(summary)

        figure, axis = plt.subplots(figsize=(10, 6))
        axis.scatter(
            plot_data["depth"],
            plot_data[column],
            alpha=0.35,
            s=28,
            color="tab:blue",
            edgecolors="none",
            label="Individual values",
        )
        axis.plot(
            by_depth["depth"],
            by_depth["mean"],
            marker="o",
            linewidth=2,
            color="tab:red",
            label="Mean",
        )
        axis.plot(
            by_depth["depth"],
            by_depth["median"],
            marker="s",
            linewidth=2,
            linestyle="--",
            color="tab:green",
            label="Median",
        )

        axis.set_title(column.replace("_", " ").title())
        axis.set_xlabel("Depth")
        axis.set_ylabel(column)
        axis.set_ylim(0, 1.5)
        axis.grid(True, alpha=0.25)
        axis.legend()
        axis.set_xticks(sorted(plot_data["depth"].unique()))
        figure.tight_layout()

        output_path = output_dir / f"scatter_{column}_by_depth.png"
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        saved_charts.append(output_path)
        print(f"Saved {output_path}")

    if not summary_frames:
        raise ValueError("No numeric ratio data was available to summarize.")

    summary_path = output_dir / SUMMARY_FILENAME
    combined_summary = pd.concat(summary_frames, ignore_index=True)
    combined_summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    return saved_charts


if __name__ == "__main__":
    create_ratio_charts()

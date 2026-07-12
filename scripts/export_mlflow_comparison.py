from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd


DEFAULT_METRICS = [
    "best_accuracy",
    "test_accuracy",
    "train_loss",
    "model_size_mb",
    "wall_clock_seconds",
    "latency_ms_per_sample",
    "peak_ram_mb",
    "peak_vram_mb",
]

DEFAULT_PARAMS = [
    "experiment_id",
    "model",
    "role",
    "dataset",
    "seed",
    "epochs",
    "batch_size",
    "optimizer",
    "device",
    "parameter_count",
    "git_commit",
]


def export_runs(
    tracking_uri: str,
    output_path: Path,
) -> pd.DataFrame:
    mlflow.set_tracking_uri(tracking_uri)

    experiments = mlflow.search_experiments()

    if not experiments:
        raise RuntimeError("No MLflow experiments were found.")

    experiment_ids = [
        experiment.experiment_id
        for experiment in experiments
        if experiment.lifecycle_stage == "active"
    ]

    runs = mlflow.search_runs(
        experiment_ids=experiment_ids,
        output_format="pandas",
        order_by=["start_time ASC"],
    )

    if runs.empty:
        raise RuntimeError("No MLflow runs were found.")

    output = pd.DataFrame()

    output["run_id"] = runs["run_id"]
    output["run_name"] = runs.get("tags.mlflow.runName")
    output["status"] = runs["status"]
    output["start_time"] = runs["start_time"]

    for param in DEFAULT_PARAMS:
        column = f"params.{param}"

        if column in runs.columns:
            output[param] = runs[column]

    for metric in DEFAULT_METRICS:
        column = f"metrics.{metric}"

        if column in runs.columns:
            output[metric] = pd.to_numeric(
                runs[column],
                errors="coerce",
            )

    # Prefer best_accuracy; use test_accuracy only when unavailable.
    if "best_accuracy" not in output.columns:
        output["best_accuracy"] = pd.NA

    if "test_accuracy" in output.columns:
        output["comparison_accuracy"] = output[
            "best_accuracy"
        ].fillna(output["test_accuracy"])
    else:
        output["comparison_accuracy"] = output[
            "best_accuracy"
        ]

    # Descriptive efficiency measures.
    if {
        "comparison_accuracy",
        "model_size_mb",
    }.issubset(output.columns):
        output["accuracy_per_mb"] = (
            output["comparison_accuracy"]
            / output["model_size_mb"]
        )

    if {
        "comparison_accuracy",
        "parameter_count",
    }.issubset(output.columns):
        output["parameter_count"] = pd.to_numeric(
            output["parameter_count"],
            errors="coerce",
        )

        output["accuracy_per_million_parameters"] = (
            output["comparison_accuracy"]
            / (output["parameter_count"] / 1_000_000)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(
        output.to_markdown(index=False),
        encoding="utf-8",
    )

    print(f"Exported {len(output)} runs")
    print(f"CSV: {output_path}")
    print(f"Markdown: {markdown_path}")

    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracking-uri",
        default="http://127.0.0.1:5050",
    )
    parser.add_argument(
        "--output",
        default="reports/comparison/mlflow_runs.csv",
    )
    args = parser.parse_args()

    export_runs(
        tracking_uri=args.tracking_uri,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()

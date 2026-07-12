from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REQUIRED_COLUMNS = {
    "role",
    "seed",
    "comparison_accuracy",
}


def effect_size_cohens_d(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first_variance = np.var(first, ddof=1)
    second_variance = np.var(second, ddof=1)

    pooled_variance = (
        (len(first) - 1) * first_variance
        + (len(second) - 1) * second_variance
    ) / (len(first) + len(second) - 2)

    if pooled_variance <= 0:
        return 0.0

    return float(
        (np.mean(first) - np.mean(second))
        / np.sqrt(pooled_variance)
    )


def confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")

    mean = np.mean(values)
    standard_error = stats.sem(values)

    interval = stats.t.interval(
        confidence,
        df=len(values) - 1,
        loc=mean,
        scale=standard_error,
    )

    return float(interval[0]), float(interval[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="reports/comparison/mlflow_runs.csv",
    )
    parser.add_argument(
        "--metric",
        default="comparison_accuracy",
    )
    parser.add_argument(
        "--output",
        default=(
            "reports/statistics/"
            "pairwise_comparison.csv"
        ),
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input)

    missing = REQUIRED_COLUMNS.difference(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    data[args.metric] = pd.to_numeric(
        data[args.metric],
        errors="coerce",
    )

    data["seed"] = pd.to_numeric(
        data["seed"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["role", "seed", args.metric]
    )

    descriptive_records: list[dict] = []

    for role, group in data.groupby("role"):
        values = group[args.metric].to_numpy(
            dtype=float
        )

        lower, upper = confidence_interval(values)

        descriptive_records.append(
            {
                "role": role,
                "n": len(values),
                "mean": float(np.mean(values)),
                "std": (
                    float(np.std(values, ddof=1))
                    if len(values) > 1
                    else float("nan")
                ),
                "median": float(np.median(values)),
                "ci95_lower": lower,
                "ci95_upper": upper,
            }
        )

    roles = sorted(data["role"].unique())
    pairwise_records: list[dict] = []

    for first_role, second_role in itertools.combinations(
        roles,
        2,
    ):
        first = data[
            data["role"] == first_role
        ][["seed", args.metric]].rename(
            columns={args.metric: "first_value"}
        )

        second = data[
            data["role"] == second_role
        ][["seed", args.metric]].rename(
            columns={args.metric: "second_value"}
        )

        paired = first.merge(
            second,
            on="seed",
            how="inner",
        )

        if len(paired) < 3:
            pairwise_records.append(
                {
                    "first_role": first_role,
                    "second_role": second_role,
                    "paired_seeds": len(paired),
                    "test": "insufficient_data",
                    "statistic": float("nan"),
                    "p_value": float("nan"),
                    "cohens_d": float("nan"),
                }
            )
            continue

        first_values = paired[
            "first_value"
        ].to_numpy(dtype=float)

        second_values = paired[
            "second_value"
        ].to_numpy(dtype=float)

        differences = first_values - second_values

        if len(differences) >= 3:
            normality_p = stats.shapiro(
                differences
            ).pvalue
        else:
            normality_p = 0.0

        if normality_p >= 0.05:
            test_name = "paired_t_test"
            test_result = stats.ttest_rel(
                first_values,
                second_values,
            )
        else:
            test_name = "wilcoxon_signed_rank"

            if np.allclose(differences, 0):
                statistic = 0.0
                p_value = 1.0
                test_result = None
            else:
                test_result = stats.wilcoxon(
                    first_values,
                    second_values,
                )

        if test_result is not None:
            statistic = float(test_result.statistic)
            p_value = float(test_result.pvalue)

        pairwise_records.append(
            {
                "first_role": first_role,
                "second_role": second_role,
                "paired_seeds": len(paired),
                "mean_first": float(
                    np.mean(first_values)
                ),
                "mean_second": float(
                    np.mean(second_values)
                ),
                "mean_difference": float(
                    np.mean(differences)
                ),
                "normality_p": float(normality_p),
                "test": test_name,
                "statistic": statistic,
                "p_value": p_value,
                "cohens_d": effect_size_cohens_d(
                    first_values,
                    second_values,
                ),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairwise = pd.DataFrame(pairwise_records)
    descriptive = pd.DataFrame(descriptive_records)

    pairwise.to_csv(output_path, index=False)

    descriptive_path = output_path.with_name(
        "descriptive_statistics.csv"
    )
    descriptive.to_csv(
        descriptive_path,
        index=False,
    )

    summary = {
        "metric": args.metric,
        "roles": roles,
        "number_of_runs": len(data),
        "minimum_recommended_seeds": 5,
    }

    summary_path = output_path.with_name(
        "statistical_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(descriptive.to_string(index=False))
    print()
    print(pairwise.to_string(index=False))
    print()
    print(f"Saved: {output_path}")
    print(f"Saved: {descriptive_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()

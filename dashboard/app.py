from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


DATA_PATH = Path(
    "reports/comparison/mlflow_runs.csv"
)

st.set_page_config(
    page_title="Architecture Pruning Research Dashboard",
    layout="wide",
)

st.title("Architecture Pruning Experiment Dashboard")
st.caption(
    "Student, teacher, pruned, KD, quantized, healed, "
    "and hybrid model comparison."
)

if not DATA_PATH.exists():
    st.error(
        "Comparison data was not found. Run "
        "`scripts/export_mlflow_comparison.py` first."
    )
    st.stop()

data = pd.read_csv(DATA_PATH)

completed = data[
    data["status"].astype(str).str.upper() == "FINISHED"
].copy()

if completed.empty:
    st.warning("No completed MLflow runs were found.")
    st.stop()

role_options = sorted(
    completed["role"].dropna().unique().tolist()
)

selected_roles = st.multiselect(
    "Model roles",
    options=role_options,
    default=role_options,
)

filtered = completed.copy()

if selected_roles:
    filtered = filtered[
        filtered["role"].isin(selected_roles)
    ]

metric_columns = [
    column
    for column in [
        "comparison_accuracy",
        "model_size_mb",
        "parameter_count",
        "latency_ms_per_sample",
        "peak_ram_mb",
        "peak_vram_mb",
        "wall_clock_seconds",
        "accuracy_per_mb",
        "accuracy_per_million_parameters",
    ]
    if column in filtered.columns
]

display_columns = [
    column
    for column in [
        "experiment_id",
        "run_name",
        "model",
        "role",
        "seed",
        *metric_columns,
    ]
    if column in filtered.columns
]

st.subheader("Experiment comparison")

st.dataframe(
    filtered[display_columns].sort_values(
        by="comparison_accuracy",
        ascending=False,
    ),
    use_container_width=True,
    hide_index=True,
)

left, middle, right = st.columns(3)

with left:
    if "comparison_accuracy" in filtered.columns:
        st.metric(
            "Highest accuracy",
            f"{filtered['comparison_accuracy'].max():.2f}%",
        )

with middle:
    if "model_size_mb" in filtered.columns:
        st.metric(
            "Smallest model",
            f"{filtered['model_size_mb'].min():.2f} MB",
        )

with right:
    if "latency_ms_per_sample" in filtered.columns:
        st.metric(
            "Lowest latency",
            f"{filtered['latency_ms_per_sample'].min():.4f} ms",
        )

chart_label = st.selectbox(
    "Chart metric",
    options=metric_columns,
)

chart_data = (
    filtered[
        ["experiment_id", chart_label]
    ]
    .dropna()
    .set_index("experiment_id")
)

st.bar_chart(chart_data)

if {
    "comparison_accuracy",
    "model_size_mb",
}.issubset(filtered.columns):
    st.subheader("Accuracy versus model size")

    scatter_data = filtered[
        [
            "experiment_id",
            "comparison_accuracy",
            "model_size_mb",
        ]
    ].dropna()

    st.scatter_chart(
        scatter_data,
        x="model_size_mb",
        y="comparison_accuracy",
    )

if "accuracy_per_mb" in filtered.columns:
    st.subheader("Capability efficiency proxy")

    efficiency = (
        filtered[
            ["experiment_id", "accuracy_per_mb"]
        ]
        .dropna()
        .sort_values(
            "accuracy_per_mb",
            ascending=False,
        )
        .set_index("experiment_id")
    )

    st.bar_chart(efficiency)

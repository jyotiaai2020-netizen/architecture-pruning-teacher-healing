from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


DATA_PATH = Path("reports/comparison/mlflow_runs.csv")
RAW_DATA_URL = (
    "https://raw.githubusercontent.com/"
    "jyotiaai2020-netizen/architecture-pruning-teacher-healing/"
    "main/reports/comparison/mlflow_runs.csv"
)

E1_RESULTS = {
    "fp32_accuracy": 93.81,
    "int8_accuracy": 93.82,
    "fp32_size_mib": 42.718,
    "int8_size_mib": 10.753,
    "compression_ratio": 3.973,
    "storage_reduction_percent": 74.83,
}

st.set_page_config(
    page_title="Teacher-Healed Quantization Research Dashboard",
    layout="wide",
)

st.title("Teacher-Healed Quantization Research Dashboard")
st.caption(
    "Evidence-first comparison of baseline, compressed, quantized, "
    "distilled, and Teacher-Healed models."
)

st.info(
    "E1 status: closure validation is in progress. Deployable-size "
    "reduction and aggregate clean-test accuracy are verified; "
    "class-level quality, calibration, paired statistics, robustness, "
    "controlled performance, energy, and EDF remain pending."
)

st.subheader("E1 — FP32 versus static INT8 QDQ")

accuracy_col, size_col, reduction_col, freeze_col = st.columns(4)

with accuracy_col:
    st.metric(
        "Clean-test accuracy",
        f"{E1_RESULTS['int8_accuracy']:.2f}% INT8",
        delta=(
            f"{E1_RESULTS['int8_accuracy'] - E1_RESULTS['fp32_accuracy']:+.2f} "
            "percentage points vs FP32"
        ),
    )

with size_col:
    st.metric(
        "INT8 deployable size",
        f"{E1_RESULTS['int8_size_mib']:.3f} MiB",
        delta=f"{E1_RESULTS['compression_ratio']:.3f}× smaller",
        delta_color="normal",
    )

with reduction_col:
    st.metric(
        "Storage reduction",
        f"{E1_RESULTS['storage_reduction_percent']:.2f}%",
        delta=f"FP32: {E1_RESULTS['fp32_size_mib']:.3f} MiB",
    )

with freeze_col:
    st.metric("Formal E1 freeze", "Not ready", delta="8 gates remaining")

e1_status = pd.DataFrame(
    [
        ("Complete deployable artifact size", "Verified"),
        ("Aggregate CIFAR-10 clean accuracy", "Provisionally supported"),
        ("Paired predictions and statistical tests", "Pending"),
        ("Per-class precision, recall, F1, and calibration", "Pending"),
        ("CIFAR-10-C robustness", "Pending"),
        ("Controlled CPU latency, throughput, and memory", "Pending"),
        ("Energy availability and EDF eligibility", "Pending"),
        ("Governance approval and formal freeze", "Pending"),
    ],
    columns=["Validation item", "Status"],
)

with st.expander("View E1 validation status", expanded=True):
    st.dataframe(e1_status, use_container_width=True, hide_index=True)
    st.caption(
        "The +0.01 percentage-point observed INT8 accuracy difference is "
        "not evidence of superiority. Quality preservation remains open "
        "until paired and class-level analyses are complete."
    )

st.divider()

if not DATA_PATH.exists():
    st.error(
        "Comparison data was not found. Run "
        "`scripts/export_mlflow_comparison.py` first."
    )
    st.markdown(f"[Download the repository CSV directly]({RAW_DATA_URL})")
    st.stop()

data = pd.read_csv(DATA_PATH)

st.subheader("Raw experiment dataset")
st.caption(
    "The download preserves every exported MLflow row and column, "
    "including incomplete runs. Dashboard filters do not change the file."
)
st.download_button(
    label="Download raw experiment data (CSV)",
    data=data.to_csv(index=False).encode("utf-8"),
    file_name="teacher_healed_quantization_raw_mlflow_runs.csv",
    mime="text/csv",
    use_container_width=True,
)
st.markdown(f"[Open the raw CSV on GitHub]({RAW_DATA_URL})")

completed = data[
    data["status"].astype(str).str.upper() == "FINISHED"
].copy()

if completed.empty:
    st.warning("No completed MLflow runs were found.")
    st.stop()

role_options = sorted(completed["role"].dropna().unique().tolist())

selected_roles = st.multiselect(
    "Model roles",
    options=role_options,
    default=role_options,
)

filtered = completed.copy()

if selected_roles:
    filtered = filtered[filtered["role"].isin(selected_roles)]

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

st.subheader("Completed experiment comparison")

sort_column = (
    "comparison_accuracy"
    if "comparison_accuracy" in filtered.columns
    else display_columns[0]
)

st.dataframe(
    filtered[display_columns].sort_values(
        by=sort_column,
        ascending=False,
    ),
    use_container_width=True,
    hide_index=True,
)

left, middle, right = st.columns(3)

with left:
    if "comparison_accuracy" in filtered.columns:
        st.metric(
            "Highest observed accuracy",
            f"{filtered['comparison_accuracy'].max():.2f}%",
        )

with middle:
    if "model_size_mb" in filtered.columns:
        st.metric(
            "Smallest serialized model",
            f"{filtered['model_size_mb'].min():.2f} MB",
        )

with right:
    if "latency_ms_per_sample" in filtered.columns:
        st.metric(
            "Lowest observed latency",
            f"{filtered['latency_ms_per_sample'].min():.4f} ms",
        )

if metric_columns:
    chart_label = st.selectbox("Chart metric", options=metric_columns)

    chart_data = (
        filtered[["experiment_id", chart_label]]
        .dropna()
        .set_index("experiment_id")
    )

    st.bar_chart(chart_data)

if {"comparison_accuracy", "model_size_mb"}.issubset(filtered.columns):
    st.subheader("Observed accuracy versus serialized model size")

    scatter_data = filtered[
        ["experiment_id", "comparison_accuracy", "model_size_mb"]
    ].dropna()

    st.scatter_chart(
        scatter_data,
        x="model_size_mb",
        y="comparison_accuracy",
    )

if "accuracy_per_mb" in filtered.columns:
    st.subheader("Accuracy-per-MB descriptive proxy")
    st.warning(
        "Accuracy per MB is a descriptive proxy only. It is not EDF or "
        "Capability Efficiency and must not be used as the E1 freeze decision."
    )

    efficiency = (
        filtered[["experiment_id", "accuracy_per_mb"]]
        .dropna()
        .sort_values("accuracy_per_mb", ascending=False)
        .set_index("experiment_id")
    )

    st.bar_chart(efficiency)

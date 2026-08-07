import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "e1b_onnx_runtime_static_int8",
    ROOT / "scripts" / "e1b_onnx_runtime_static_int8.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_wilson_confidence_interval_is_reasonable() -> None:
    lower, upper = MODULE.wilson_confidence_interval(950, 1000)
    assert lower < upper
    assert lower < 0.96
    assert upper > 0.96


def test_summarize_per_class_metrics_reports_all_classes() -> None:
    labels = [0, 0, 1, 1, 2, 2]
    predictions = [0, 1, 1, 2, 2, 2]
    summary = MODULE.summarize_per_class_metrics(labels, predictions, ["cat", "dog", "bird"])

    assert summary["cat"]["support"] == 2
    assert summary["cat"]["accuracy"] == 0.5
    assert summary["dog"]["support"] == 2
    assert summary["dog"]["precision"] == 0.5
    assert summary["bird"]["recall"] == 1.0

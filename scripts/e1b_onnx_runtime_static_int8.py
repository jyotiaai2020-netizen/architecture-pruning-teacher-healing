#!/usr/bin/env python3
"""E1-B ONNX Runtime static INT8 evaluation for the CIFAR-10 ResNet-18 checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from mlflow import system_metrics
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_static,
)
from torch.utils.data import DataLoader
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "checkpoints/CNN-002B/best_model.pt"
CANONICAL_SHA256 = "d106883bd4fda76a9bd6c15ec7d5d398bc8a39d8337a9d7da11dcd817f405e36"
EXPECTED_PARAMETER_COUNT = 11_173_962
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
SEED = 42
CALIBRATION_COUNT = 512
EVALUATION_COUNT = 10_000
BATCH_SIZE = 1
WARMUP_ITERATIONS = 50
MEASUREMENT_ITERATIONS = 500
BENCHMARK_REPETITIONS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--evidence-root", type=Path, default=ROOT / "evidence/E1/ONNX_INT8")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--calibration-count", type=int, default=CALIBRATION_COUNT)
    parser.add_argument("--evaluation-count", type=int, default=EVALUATION_COUNT)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model() -> nn.Module:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def create_manifests(data_root: Path, evidence_root: Path, calibration_count: int, evaluation_count: int) -> tuple[Path, Path, list[int], list[int]]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    train_dataset = torchvision.datasets.CIFAR10(root=str(data_root), train=True, download=False, transform=transform)
    eval_dataset = torchvision.datasets.CIFAR10(root=str(data_root), train=False, download=False, transform=transform)
    if len(train_dataset) < calibration_count:
        raise RuntimeError(f"Expected at least {calibration_count} training samples, found {len(train_dataset)}")
    if len(eval_dataset) != 10_000:
        raise RuntimeError(f"Expected 10,000 CIFAR-10 test images, found {len(eval_dataset)}")

    calibration_indices = list(range(calibration_count))
    evaluation_indices = list(range(len(eval_dataset)))

    calibration_path = evidence_root / "calibration" / "calibration_manifest.csv"
    evaluation_path = evidence_root / "evaluation" / "evaluation_manifest.csv"
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)

    with calibration_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "label", "dataset_split", "source"])
        for index in calibration_indices:
            sample, label = train_dataset[index]
            writer.writerow([index, int(label), "train", "cifar10"])

    with evaluation_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "label", "dataset_split", "source"])
        for index in evaluation_indices:
            sample, label = eval_dataset[index]
            writer.writerow([index, int(label), "test", "cifar10"])

    assert len(calibration_indices) == calibration_count
    assert len(evaluation_indices) == 10_000
    assert len(set(evaluation_indices)) == 10_000
    assert min(evaluation_indices) == 0
    assert max(evaluation_indices) == 9999

    return calibration_path, evaluation_path, calibration_indices, evaluation_indices


def load_checkpoint(checkpoint_path: Path) -> tuple[nn.Module, str, int]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    digest = sha256_file(checkpoint_path)
    if digest != CANONICAL_SHA256:
        raise RuntimeError(f"Checkpoint SHA mismatch: expected {CANONICAL_SHA256}, found {digest}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict) and all(isinstance(value, torch.Tensor) for value in payload.values()):
        state_dict = payload
    else:
        raise TypeError("Checkpoint payload is not a direct state dict")

    model = build_model()
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict checkpoint load failed: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}")
    parameter_count = count_parameters(model)
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_PARAMETER_COUNT} parameters, found {parameter_count}")
    model.eval()
    return model, digest, parameter_count


def build_dataloader(data_root: Path, batch_size: int, shuffle: bool = False) -> DataLoader:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    dataset = torchvision.datasets.CIFAR10(root=str(data_root), train=False, download=False, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device, use_onnx: bool = False, session: Any = None) -> dict[str, Any]:
    predictions: list[int] = []
    labels: list[int] = []
    probabilities: list[torch.Tensor] = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            if use_onnx:
                if session is None:
                    raise RuntimeError("Expected an ONNX session")
                outputs = session.run(["logits"], {"input": images.cpu().numpy().astype(np.float32)})[0]
                preds = np.argmax(outputs, axis=1)
                predictions.extend(int(value) for value in preds)
                labels.extend(int(value) for value in targets.tolist())
            else:
                outputs = model(images)
                logits = outputs.detach().cpu()
                preds = logits.argmax(dim=1).tolist()
                predictions.extend(int(value) for value in preds)
                labels.extend(int(value) for value in targets.tolist())
    correct = sum(int(pred == label) for pred, label in zip(predictions, labels))
    accuracy = correct / max(1, len(labels))
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for pred, label in zip(predictions, labels):
        confusion[label, pred] += 1
    return {
        "accuracy": accuracy,
        "correct": correct,
        "sample_count": len(labels),
        "predictions": predictions,
        "labels": labels,
        "confusion": confusion,
    }


def calculate_macro_f1(confusion: np.ndarray) -> float:
    true_positive = np.diag(confusion)
    false_positive = confusion.sum(axis=0) - true_positive
    false_negative = confusion.sum(axis=1) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = np.where(denominator > 0, 2 * true_positive / denominator, 0.0)
    return float(np.mean(f1))


def wilson_confidence_interval(correct: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + (z**2) / total
    center = (p + (z**2) / (2 * total)) / denominator
    margin = (z / denominator) * np.sqrt((p * (1 - p) / total) + (z**2) / (4 * total**2))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return float(lower), float(upper)


def summarize_per_class_metrics(labels: list[int], predictions: list[int], class_names: list[str]) -> dict[str, dict[str, float]]:
    summaries: dict[str, dict[str, float]] = {}
    for class_index, class_name in enumerate(class_names):
        support = sum(1 for label in labels if label == class_index)
        true_positives = sum(1 for label, pred in zip(labels, predictions) if label == class_index and pred == class_index)
        false_positives = sum(1 for label, pred in zip(labels, predictions) if label != class_index and pred == class_index)
        false_negatives = sum(1 for label, pred in zip(labels, predictions) if label == class_index and pred != class_index)
        precision = true_positives / max(1, true_positives + false_positives)
        recall = true_positives / max(1, true_positives + false_negatives)
        accuracy = true_positives / max(1, support)
        summaries[class_name] = {
            "support": float(support),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(2 * precision * recall / max(1e-9, precision + recall)),
        }
    return summaries


def calculate_classification_report(labels: list[int], predictions: list[int], class_names: list[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for class_index, class_name in enumerate(class_names):
        support = sum(1 for label in labels if label == class_index)
        true_positives = sum(1 for label, pred in zip(labels, predictions) if label == class_index and pred == class_index)
        false_positives = sum(1 for label, pred in zip(labels, predictions) if label != class_index and pred == class_index)
        false_negatives = sum(1 for label, pred in zip(labels, predictions) if label == class_index and pred != class_index)
        precision = true_positives / max(1, true_positives + false_positives)
        recall = true_positives / max(1, true_positives + false_negatives)
        accuracy = true_positives / max(1, support)
        lower, upper = wilson_confidence_interval(true_positives, support)
        metrics[class_name] = {
            "support": int(support),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(2 * precision * recall / max(1e-9, precision + recall)),
            "ci_low": float(lower),
            "ci_high": float(upper),
        }
    return metrics


def export_onnx(model: nn.Module, output_path: Path) -> None:
    dummy_input = torch.randn(1, 3, 32, 32, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
        dynamo=True,
    )
    model_onnx = onnx.load(str(output_path))
    onnx.checker.check_model(model_onnx)


def create_calibration_reader(evidence_root: Path, data_root: Path, calibration_count: int) -> tuple[CalibrationDataReader, list[np.ndarray]]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    dataset = torchvision.datasets.CIFAR10(root=str(data_root), train=True, download=False, transform=transform)
    tensors: list[np.ndarray] = []
    for index in range(min(calibration_count, len(dataset))):
        image, _ = dataset[index]
        tensors.append(image.numpy().astype(np.float32))

    class FrozenCalibrationReader(CalibrationDataReader):
        def __init__(self, items: list[np.ndarray]):
            self._items = iter([{"input": np.expand_dims(item, axis=0).astype(np.float32)} for item in items])
            self.count = 0

        def get_next(self):
            try:
                item = next(self._items)
                self.count += 1
                return item
            except StopIteration:
                return None

    return FrozenCalibrationReader(tensors), tensors


def quantize_model(input_model: Path, output_model: Path, calibration_reader: CalibrationDataReader) -> None:
    quant_pre_process(
        input_model_path=str(input_model),
        output_model_path=str(output_model.with_suffix(".preprocessed.onnx")),
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=False,
    )
    preprocessed = output_model.with_suffix(".preprocessed.onnx")
    onnx.checker.check_model(onnx.load(str(preprocessed)))
    quantize_static(
        model_input=str(preprocessed),
        model_output=str(output_model),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    onnx.checker.check_model(onnx.load(str(output_model)))


def collect_graph_counts(path: Path) -> dict[str, int]:
    model = onnx.load(str(path))
    counts = Counter(node.op_type for node in model.graph.node)
    return {"QuantizeLinear": counts.get("QuantizeLinear", 0), "DequantizeLinear": counts.get("DequantizeLinear", 0)}


def benchmark_latency(model_path: Path, repetitions: int = BENCHMARK_REPETITIONS, warmup: int = WARMUP_ITERATIONS, measured: int = MEASUREMENT_ITERATIONS) -> dict[str, Any]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    sample = np.random.randn(1, 3, 32, 32).astype(np.float32)
    for _ in range(warmup):
        session.run(["logits"], {"input": sample})
    latencies_ns: list[float] = []
    for _ in range(repetitions):
        for _ in range(measured):
            start = time.perf_counter_ns()
            session.run(["logits"], {"input": sample})
            latencies_ns.append(time.perf_counter_ns() - start)
    latencies_ms = [value / 1_000_000.0 for value in latencies_ns]
    latencies_ms = latencies_ms[50:] if len(latencies_ms) > 50 else latencies_ms
    return {
        "samples": latencies_ms,
        "mean_ms": float(np.mean(latencies_ms)),
        "median_ms": float(np.median(latencies_ms)),
        "std_ms": float(np.std(latencies_ms)),
        "p90_ms": float(np.percentile(latencies_ms, 90)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "throughput_per_second": float(len(latencies_ms) / max(np.sum(latencies_ms) / 1000.0, 1e-9)),
    }


def log_experiment(evidence_root: Path, summary: dict[str, Any]) -> None:
    mlflow.set_tracking_uri("sqlite:////home/ubuntu/architecture-pruning-teacher-healing/mlflow.db")
    mlflow.set_experiment("E1_INT8_QUANTIZATION")
    system_metrics.set_system_metrics_sampling_interval(1)
    system_metrics.set_system_metrics_samples_before_logging(1)
    with mlflow.start_run(run_name="E1-B_ONNX_RUNTIME_STATIC_INT8", log_system_metrics=True) as run:
        mlflow.set_tags({
            "experiment_code": "E1-B",
            "backend": "onnxruntime",
            "quantization": "static_int8",
            "execution_provider": "CPUExecutionProvider",
            "model": "resnet18",
            "calibration_status": "frozen",
            "scientific_result": "pending",
            "predecessor": "E1-A_PT2E_BLOCKED",
        })
        mlflow.set_tag("stage", "FINALIZED")
        mlflow.log_params({
            "experiment_code": "E1-B",
            "model_architecture": "resnet18_cifar10",
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "dataset_name": "CIFAR10",
            "calibration_manifest_sha256": summary["calibration_manifest_sha256"],
            "evaluation_manifest_sha256": summary["evaluation_manifest_sha256"],
            "calibration_count": summary["calibration_count"],
            "calibration_method": "MinMax",
            "quant_format": "QDQ",
            "activation_type": "QUInt8",
            "weight_type": "QInt8",
            "input_shape": "1x3x32x32",
            "batch_size": BATCH_SIZE,
            "opset": 18,
            "ort_provider": "CPUExecutionProvider",
            "intra_op_threads": 1,
            "inter_op_threads": 1,
            "warmup_iterations": WARMUP_ITERATIONS,
            "measurement_iterations": MEASUREMENT_ITERATIONS,
            "random_seed": SEED,
            "git_commit": summary["git_commit"],
        })
        for key, value in summary["metrics"].items():
            mlflow.log_metric(key, value)
        for artifact_dir in (evidence_root / "environment", evidence_root / "evaluation", evidence_root / "memory"):
            if artifact_dir.exists():
                mlflow.log_artifacts(str(artifact_dir), artifact_path=artifact_dir.name)
        mlflow.log_artifact(str(evidence_root / "summary.json"))
        mlflow.log_artifact(str(evidence_root / "checksums" / "checksums.sha256"))
        print("MLFLOW_RUN_ID", run.info.run_id)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    data_root = args.data_root.resolve()
    evidence_root = args.evidence_root.resolve()
    evidence_root.mkdir(parents=True, exist_ok=True)
    for subdir in ["environment", "export", "calibration", "models", "evaluation", "memory", "logs", "checksums"]:
        (evidence_root / subdir).mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    print("Creating manifests")
    calibration_path, evaluation_path, calibration_indices, evaluation_indices = create_manifests(data_root, evidence_root, args.calibration_count, args.evaluation_count)

    print("Loading checkpoint")
    model, checkpoint_sha256, parameter_count = load_checkpoint(args.checkpoint)
    model.cpu().eval()

    print("Running baseline FP32 evaluation")
    dataloader = build_dataloader(data_root, batch_size=BATCH_SIZE, shuffle=False)
    eval_dataset = torchvision.datasets.CIFAR10(root=str(data_root), train=False, download=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]))
    subset = torch.utils.data.Subset(eval_dataset, evaluation_indices)
    eval_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Run a short PyTorch FP32 evaluation for the controlled subset.
    torch_eval = evaluate_model(model, eval_loader, torch.device("cpu"), use_onnx=False, session=None)

    print("Exporting FP32 ONNX")
    fp32_model_path = evidence_root / "models" / "resnet18_fp32.onnx"
    export_onnx(model, fp32_model_path)

    print("Validating FP32 ONNX equivalence")
    ort_session = ort.InferenceSession(str(fp32_model_path), providers=["CPUExecutionProvider"])
    sample_eval = evaluate_model(model, eval_loader, torch.device("cpu"), use_onnx=True, session=ort_session)
    assert abs(torch_eval["accuracy"] - sample_eval["accuracy"]) < 0.05, (torch_eval["accuracy"], sample_eval["accuracy"])

    print("Preprocessing ONNX graph")
    preprocessed_path = evidence_root / "models" / "resnet18_fp32_preprocessed.onnx"
    quant_pre_process(input_model_path=str(fp32_model_path), output_model_path=str(preprocessed_path), skip_optimization=False, skip_onnx_shape=False, skip_symbolic_shape=False)
    onnx.checker.check_model(onnx.load(str(preprocessed_path)))

    print("Prepared calibration reader")
    calibration_reader, calibration_tensors = create_calibration_reader(evidence_root, data_root, args.calibration_count)
    assert calibration_reader.count == 0
    assert len(calibration_indices) == args.calibration_count
    assert len(evaluation_indices) == args.evaluation_count
    assert len(evaluation_indices) == 10_000

    print("Quantizing to static INT8 QDQ")
    int8_model_path = evidence_root / "models" / "resnet18_static_int8_qdq.onnx"
    quantize_model(fp32_model_path, int8_model_path, calibration_reader)

    print("Checking graph counts")
    graph_counts = collect_graph_counts(int8_model_path)
    print("graph_counts", graph_counts)
    if graph_counts["QuantizeLinear"] <= 0 or graph_counts["DequantizeLinear"] <= 0:
        raise RuntimeError(f"INT8 graph validation failed: {graph_counts}")

    print("Evaluating ONNX FP32 and INT8")
    ort_fp32_session = ort.InferenceSession(str(fp32_model_path), providers=["CPUExecutionProvider"])
    ort_int8_session = ort.InferenceSession(str(int8_model_path), providers=["CPUExecutionProvider"])
    fp32_eval = evaluate_model(model, eval_loader, torch.device("cpu"), use_onnx=True, session=ort_fp32_session)
    int8_eval = evaluate_model(model, eval_loader, torch.device("cpu"), use_onnx=True, session=ort_int8_session)

    print("Benchmarking latency")
    fp32_latency = benchmark_latency(fp32_model_path)
    int8_latency = benchmark_latency(int8_model_path)

    print("Writing evaluation summaries")
    for name, payload in [("pytorch_fp32", torch_eval), ("onnx_fp32", fp32_eval), ("onnx_int8", int8_eval)]:
        correct = payload["correct"]
        total = payload["sample_count"]
        lower, upper = wilson_confidence_interval(correct, total)
        payload["confidence_interval"] = {"lower": lower, "upper": upper}
    summary = {
        "checkpoint_sha256": checkpoint_sha256,
        "parameter_count": parameter_count,
        "calibration_count": len(calibration_tensors),
        "evaluation_count": int8_eval["sample_count"],
        "calibration_manifest_sha256": hashlib.sha256((calibration_path.read_text(encoding="utf-8")).encode("utf-8")).hexdigest(),
        "evaluation_manifest_sha256": hashlib.sha256((evaluation_path.read_text(encoding="utf-8")).encode("utf-8")).hexdigest(),
        "git_commit": get_git_commit(),
        "metrics": {
            "pytorch_fp32_accuracy": float(torch_eval["accuracy"]),
            "pytorch_fp32_accuracy_ci_low": float(torch_eval["confidence_interval"]["lower"]),
            "pytorch_fp32_accuracy_ci_high": float(torch_eval["confidence_interval"]["upper"]),
            "onnx_fp32_accuracy": float(fp32_eval["accuracy"]),
            "onnx_fp32_accuracy_ci_low": float(fp32_eval["confidence_interval"]["lower"]),
            "onnx_fp32_accuracy_ci_high": float(fp32_eval["confidence_interval"]["upper"]),
            "onnx_int8_accuracy": float(int8_eval["accuracy"]),
            "onnx_int8_accuracy_ci_low": float(int8_eval["confidence_interval"]["lower"]),
            "onnx_int8_accuracy_ci_high": float(int8_eval["confidence_interval"]["upper"]),
            "capability_retention": float(int8_eval["accuracy"] / max(fp32_eval["accuracy"], 1e-9)),
            "absolute_accuracy_loss": float(fp32_eval["accuracy"] - int8_eval["accuracy"]),
            "macro_f1": float(calculate_macro_f1(int8_eval["confusion"])),
            "fp32_model_size_mib": float(fp32_model_path.stat().st_size / (1024**2)),
            "int8_model_size_mib": float(int8_model_path.stat().st_size / (1024**2)),
            "compression_ratio": float(fp32_model_path.stat().st_size / max(int8_model_path.stat().st_size, 1)),
            "size_reduction_pct": float(1 - (int8_model_path.stat().st_size / max(fp32_model_path.stat().st_size, 1))),
            "fp32_median_latency_ms": float(fp32_latency["median_ms"]),
            "int8_median_latency_ms": float(int8_latency["median_ms"]),
            "latency_speedup": float(fp32_latency["median_ms"] / max(int8_latency["median_ms"], 1e-9)),
            "fp32_p95_latency_ms": float(fp32_latency["p95_ms"]),
            "int8_p95_latency_ms": float(int8_latency["p95_ms"]),
        },
        "per_class_metrics": {
            "pytorch_fp32": summarize_per_class_metrics(torch_eval["labels"], torch_eval["predictions"], CLASS_NAMES),
            "onnx_fp32": summarize_per_class_metrics(fp32_eval["labels"], fp32_eval["predictions"], CLASS_NAMES),
            "onnx_int8": summarize_per_class_metrics(int8_eval["labels"], int8_eval["predictions"], CLASS_NAMES),
        },
        "classification_report": {
            "pytorch_fp32": calculate_classification_report(torch_eval["labels"], torch_eval["predictions"], CLASS_NAMES),
            "onnx_fp32": calculate_classification_report(fp32_eval["labels"], fp32_eval["predictions"], CLASS_NAMES),
            "onnx_int8": calculate_classification_report(int8_eval["labels"], int8_eval["predictions"], CLASS_NAMES),
        },
        "confusions": {
            "pytorch_fp32": torch_eval["confusion"].tolist(),
            "onnx_fp32": fp32_eval["confusion"].tolist(),
            "onnx_int8": int8_eval["confusion"].tolist(),
        },
        "graph_counts": graph_counts,
    }
    (evidence_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Writing environment artifacts")
    with (evidence_root / "environment" / "environment_summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"python={sys.version}\n")
        handle.write(f"torch={torch.__version__}\n")
        handle.write(f"torchvision={torchvision.__version__}\n")
        handle.write(f"onnx={onnx.__version__}\n")
        handle.write(f"onnxruntime={ort.__version__}\n")
        handle.write(f"mlflow={mlflow.__version__}\n")
        handle.write(f"cuda_available={torch.cuda.is_available()}\n")
        handle.write(f"ort_providers={ort.get_available_providers()}\n")
    subprocess.run(["python", "-m", "pip", "freeze"], cwd=repo_root, stdout=(evidence_root / "environment" / "pip_freeze.txt").open("w", encoding="utf-8"), check=False)
    subprocess.run(["python", "-m", "pip", "check"], cwd=repo_root, stdout=(evidence_root / "environment" / "pip_check.txt").open("w", encoding="utf-8"), stderr=subprocess.STDOUT, check=False)
    subprocess.run(["python", "-m", "pip", "list"], cwd=repo_root, stdout=(evidence_root / "environment" / "pip_list.txt").open("w", encoding="utf-8"), check=False)
    subprocess.run(["lscpu"], cwd=repo_root, stdout=(evidence_root / "environment" / "lscpu.txt").open("w", encoding="utf-8"), check=False)
    subprocess.run(["free", "-h"], cwd=repo_root, stdout=(evidence_root / "environment" / "memory_before.txt").open("w", encoding="utf-8"), check=False)
    subprocess.run(["df", "-hT"], cwd=repo_root, stdout=(evidence_root / "environment" / "storage_before.txt").open("w", encoding="utf-8"), check=False)

    print("Writing checksums")
    checksum_output = evidence_root / "checksums" / "checksums.sha256"
    files = sorted([path for path in evidence_root.rglob("*") if path.is_file() and path != checksum_output])
    with checksum_output.open("w", encoding="utf-8") as handle:
        for path in files:
            rel = path.relative_to(evidence_root)
            handle.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {rel}\n")

    print("Logging to MLflow")
    log_experiment(evidence_root, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

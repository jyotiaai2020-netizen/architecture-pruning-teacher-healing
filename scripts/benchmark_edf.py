#!/usr/bin/env python3
"""Controlled E0 FP32 Effective Deployment Footprint benchmark.

This script benchmarks the validated CIFAR-10 ResNet-18 checkpoint on CPU and
creates the EDF v1 evidence package. It deliberately fails closed when the
checkpoint hash, architecture, strict state-dict load, or parameter count does
not match the frozen E0 candidate.

Energy is an explicitly labelled proxy:

    estimated joules per inference =
        assumed CPU watts * measured batch duration seconds / batch size

Example:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 python scripts/benchmark_edf.py \
      --data-root data --assumed-cpu-watts 15 \
      --energy-source "AWS host CPU TDP assumption documented by researcher"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


CANONICAL_CHECKPOINT = Path("checkpoints/CNN-002B/best_model.pt")
CANONICAL_SHA256 = (
    "d106883bd4fda76a9bd6c15ec7d5d398bc8a39d8337a9d7da11dcd817f405e36"
)
EXPECTED_PARAMETERS = 11_173_962
EXPECTED_SAMPLES = 10_000
DEFAULT_OUTPUT = Path("results/E0/edf_v1")
DEFAULT_BATCH_SIZES = (1, 8, 32, 128)
DEFAULT_SEEDS = (42, 123, 2026)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class BenchmarkConfig:
    checkpoint: str
    checkpoint_sha256: str
    output_dir: str
    data_root: str
    batch_sizes: tuple[int, ...]
    warmup_iterations: int
    timed_iterations: int
    repetitions: int
    seeds: tuple[int, ...]
    cpu_threads: int
    quantization_backend: str
    assumed_cpu_watts: float
    energy_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CANONICAL_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=list(DEFAULT_BATCH_SIZES)
    )
    parser.add_argument("--warmup-iterations", type=int, default=50)
    parser.add_argument("--timed-iterations", type=int, default=500)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--quantization-backend", default="x86")
    parser.add_argument(
        "--assumed-cpu-watts",
        type=float,
        required=True,
        help="Documented CPU-power assumption used only for the energy proxy.",
    )
    parser.add_argument(
        "--energy-source",
        required=True,
        help="Citation or researcher note supporting --assumed-cpu-watts.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CIFAR-10 when it is absent from --data-root.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = {
        "warmup_iterations": args.warmup_iterations,
        "timed_iterations": args.timed_iterations,
        "repetitions": args.repetitions,
        "cpu_threads": args.cpu_threads,
    }
    for name, value in positive_ints.items():
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
    if any(batch_size < 1 for batch_size in args.batch_sizes):
        raise ValueError("Every batch size must be at least 1")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        raise ValueError("Batch sizes must be unique")
    if not args.seeds:
        raise ValueError("At least one seed is required")
    if args.assumed_cpu_watts <= 0:
        raise ValueError("--assumed-cpu-watts must be greater than zero")
    if not args.energy_source.strip():
        raise ValueError("--energy-source cannot be blank")
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {args.output_dir}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_runtime(cpu_threads: int, backend: str) -> None:
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        configured = os.environ.get(variable)
        if configured != str(cpu_threads):
            raise RuntimeError(
                f"{variable} must equal {cpu_threads}; current value is {configured!r}"
            )

    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError(
            f"Unsupported quantization backend {backend!r}; supported engines: "
            f"{torch.backends.quantized.supported_engines}"
        )
    torch.backends.quantized.engine = backend
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    if torch.get_num_threads() != cpu_threads:
        raise RuntimeError("PyTorch did not apply the requested CPU thread count")
    if torch.get_num_interop_threads() != 1:
        raise RuntimeError("PyTorch interop threads must equal 1")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model() -> nn.Module:
    model = models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    return model


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                checkpoint = candidate
                break
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint does not contain a recognizable state dict")

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        normalized_key = str(key)
        for prefix in ("module.", "model."):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix) :]
        state_dict[normalized_key] = value
    if not state_dict:
        raise ValueError("No tensor parameters were found in the checkpoint")
    return state_dict


def load_validated_model(checkpoint_path: Path) -> tuple[nn.Module, str, int]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != CANONICAL_SHA256:
        raise RuntimeError(
            "Checkpoint SHA-256 mismatch. "
            f"Expected {CANONICAL_SHA256}, found {checkpoint_hash}"
        )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(extract_state_dict(payload), strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"Parameter-count mismatch: expected {EXPECTED_PARAMETERS}, "
            f"found {parameter_count}"
        )
    model.eval()
    return model, checkpoint_hash, parameter_count


def build_test_dataset(data_root: Path, download: bool) -> datasets.CIFAR10:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )
    dataset = datasets.CIFAR10(
        root=data_root, train=False, transform=transform, download=download
    )
    if len(dataset) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_SAMPLES} CIFAR-10 test samples, found {len(dataset)}"
        )
    return dataset


def expected_calibration_error(
    probabilities: torch.Tensor, targets: torch.Tensor, bins: int = 15
) -> float:
    confidences, predictions = probabilities.max(dim=1)
    accuracies = predictions.eq(targets)
    boundaries = torch.linspace(0.0, 1.0, bins + 1)
    ece = torch.zeros((), dtype=torch.float64)
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        selected = confidences.gt(lower) & confidences.le(upper)
        if selected.any():
            fraction = selected.double().mean()
            bin_accuracy = accuracies[selected].double().mean()
            bin_confidence = confidences[selected].double().mean()
            ece += fraction * (bin_confidence - bin_accuracy).abs()
    return float(ece)


def macro_f1_score(
    predictions: torch.Tensor, targets: torch.Tensor, class_count: int = 10
) -> float:
    scores: list[float] = []
    for class_id in range(class_count):
        predicted_class = predictions.eq(class_id)
        target_class = targets.eq(class_id)
        true_positive = (predicted_class & target_class).sum().item()
        false_positive = (predicted_class & ~target_class).sum().item()
        false_negative = (~predicted_class & target_class).sum().item()
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(mean(scores))


@torch.inference_mode()
def evaluate_capability(
    model: nn.Module, dataset: datasets.CIFAR10, seeds: Iterable[int]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_seed: list[dict[str, Any]] = []
    canonical_predictions: list[dict[str, Any]] = []

    for seed_index, seed in enumerate(seeds):
        seed_everything(seed)
        loader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        logits_parts: list[torch.Tensor] = []
        target_parts: list[torch.Tensor] = []
        for images, targets in loader:
            logits_parts.append(model(images))
            target_parts.append(targets)

        logits = torch.cat(logits_parts)
        targets = torch.cat(target_parts)
        probabilities = logits.softmax(dim=1)
        predictions = probabilities.argmax(dim=1)
        nll = nn.functional.cross_entropy(logits, targets, reduction="mean").item()
        metrics = {
            "seed": seed,
            "sample_count": len(targets),
            "clean_accuracy": float(predictions.eq(targets).double().mean()),
            "clean_accuracy_percent": float(
                100.0 * predictions.eq(targets).double().mean()
            ),
            "macro_f1": macro_f1_score(predictions, targets),
            "negative_log_likelihood": float(nll),
            "expected_calibration_error_15_bin": expected_calibration_error(
                probabilities, targets, bins=15
            ),
        }
        per_seed.append(metrics)

        if seed_index == 0:
            confidences = probabilities.max(dim=1).values
            canonical_predictions = [
                {
                    "sample_index": index,
                    "seed": seed,
                    "target": int(targets[index]),
                    "prediction": int(predictions[index]),
                    "confidence": float(confidences[index]),
                    "correct": bool(predictions[index] == targets[index]),
                }
                for index in range(len(targets))
            ]

    metric_names = (
        "clean_accuracy",
        "clean_accuracy_percent",
        "macro_f1",
        "negative_log_likelihood",
        "expected_calibration_error_15_bin",
    )
    aggregate = {
        name: float(mean(float(run[name]) for run in per_seed))
        for name in metric_names
    }
    aggregate.update(
        {
            "sample_count": EXPECTED_SAMPLES,
            "class_count": 10,
            "ece_bins": 15,
            "seeds": list(seeds),
            "per_seed": per_seed,
            "corruption_robustness": {
                "status": "not_measured_by_this_script",
                "reason": "CIFAR-10-C requires a separately verified dataset artifact.",
            },
            "failure_consistency": {
                "status": "reference_predictions_created",
                "reference": "E0",
            },
        }
    )
    return aggregate, canonical_predictions


def current_rss_mb() -> float:
    try:
        import psutil  # type: ignore

        return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    except ImportError:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024.0 if sys.platform != "darwin" else 1024.0**2
        return usage / divisor


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


@torch.inference_mode()
def benchmark_latency(
    model: nn.Module,
    dataset: datasets.CIFAR10,
    batch_sizes: Iterable[int],
    warmup_iterations: int,
    timed_iterations: int,
    repetitions: int,
    assumed_cpu_watts: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    peak_rss_mb = current_rss_mb()

    for batch_size in batch_sizes:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
        )
        iterator = iter(loader)

        def next_images() -> torch.Tensor:
            nonlocal iterator
            try:
                images, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                images, _ = next(iterator)
            return images

        for _ in range(warmup_iterations):
            model(next_images())

        batch_durations_ms: list[float] = []
        per_sample_latencies_ms: list[float] = []
        for repetition in range(1, repetitions + 1):
            for iteration in range(1, timed_iterations + 1):
                images = next_images()
                started_ns = time.perf_counter_ns()
                model(images)
                elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
                latency_per_sample_ms = elapsed_ms / batch_size
                estimated_joules_per_sample = (
                    assumed_cpu_watts * elapsed_ms / 1000.0 / batch_size
                )
                batch_durations_ms.append(elapsed_ms)
                per_sample_latencies_ms.append(latency_per_sample_ms)
                peak_rss_mb = max(peak_rss_mb, current_rss_mb())
                raw_rows.append(
                    {
                        "batch_size": batch_size,
                        "repetition": repetition,
                        "iteration": iteration,
                        "batch_latency_ms": elapsed_ms,
                        "latency_ms_per_sample": latency_per_sample_ms,
                        "throughput_samples_per_second": (
                            batch_size / (elapsed_ms / 1000.0)
                        ),
                        "estimated_joules_per_sample": estimated_joules_per_sample,
                    }
                )

        total_samples = batch_size * len(batch_durations_ms)
        total_seconds = sum(batch_durations_ms) / 1000.0
        summaries[str(batch_size)] = {
            "batch_size": batch_size,
            "measurement_count": len(batch_durations_ms),
            "sample_count": total_samples,
            "p50_latency_ms_per_sample": percentile(
                per_sample_latencies_ms, 50
            ),
            "p95_latency_ms_per_sample": percentile(
                per_sample_latencies_ms, 95
            ),
            "p99_latency_ms_per_sample": percentile(
                per_sample_latencies_ms, 99
            ),
            "mean_latency_ms_per_sample": float(mean(per_sample_latencies_ms)),
            "throughput_samples_per_second": total_samples / total_seconds,
            "timed_inference_seconds": total_seconds,
            "estimated_joules_per_inference": (
                assumed_cpu_watts * total_seconds / total_samples
            ),
        }

    return raw_rows, {
        "batch_sizes": summaries,
        "peak_cpu_ram_mb": peak_rss_mb,
        "peak_gpu_vram_mb": None,
    }


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def collect_environment(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "deployment_device": "cpu",
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime_version": torch.version.cuda,
        "quantization_backend": torch.backends.quantized.engine,
        "supported_quantization_engines": (
            torch.backends.quantized.supported_engines
        ),
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "environment_threads": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "cpu_description": command_output(["lscpu"]),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_tree_status": command_output(["git", "status", "--porcelain"]),
        "config": asdict(config),
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_manifest(
    output_dir: Path, config: BenchmarkConfig, started_at: str
) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            files[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return {
        "schema_version": "edf_v1",
        "experiment_id": "E0",
        "variant": "FP32",
        "status": "benchmark_completed_pending_protocol_freeze",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": config.checkpoint,
            "sha256": config.checkpoint_sha256,
            "role": "canonical_best_validation_checkpoint",
        },
        "energy_proxy": {
            "is_proxy": True,
            "method": "estimated_cpu_power_times_inference_duration",
            "formula": (
                "assumed_cpu_watts * measured_batch_seconds / batch_size"
            ),
            "assumed_cpu_watts": config.assumed_cpu_watts,
            "source": config.energy_source,
        },
        "files": files,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    started_at = datetime.now(timezone.utc).isoformat()
    configure_runtime(args.cpu_threads, args.quantization_backend)
    seed_everything(args.seeds[0])

    model, checkpoint_hash, parameter_count = load_validated_model(args.checkpoint)
    dataset = build_test_dataset(args.data_root, args.download)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    config = BenchmarkConfig(
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=checkpoint_hash,
        output_dir=str(args.output_dir),
        data_root=str(args.data_root),
        batch_sizes=tuple(args.batch_sizes),
        warmup_iterations=args.warmup_iterations,
        timed_iterations=args.timed_iterations,
        repetitions=args.repetitions,
        seeds=tuple(args.seeds),
        cpu_threads=args.cpu_threads,
        quantization_backend=args.quantization_backend,
        assumed_cpu_watts=args.assumed_cpu_watts,
        energy_source=args.energy_source,
    )

    capability, predictions = evaluate_capability(model, dataset, args.seeds)
    raw_latency, resources = benchmark_latency(
        model=model,
        dataset=dataset,
        batch_sizes=args.batch_sizes,
        warmup_iterations=args.warmup_iterations,
        timed_iterations=args.timed_iterations,
        repetitions=args.repetitions,
        assumed_cpu_watts=args.assumed_cpu_watts,
    )

    resources.update(
        {
            "serialized_checkpoint_size_bytes": args.checkpoint.stat().st_size,
            "serialized_checkpoint_size_mib": (
                args.checkpoint.stat().st_size / (1024**2)
            ),
            "energy": {
                "enabled": True,
                "is_proxy": True,
                "method": "estimated_cpu_power_times_inference_duration",
                "unit": "joules_per_inference",
                "assumed_cpu_watts": args.assumed_cpu_watts,
                "source": args.energy_source,
                "formula": (
                    "assumed_cpu_watts * measured_batch_seconds / batch_size"
                ),
            },
        }
    )
    environment = collect_environment(config)
    summary = {
        "schema_version": "edf_v1",
        "experiment_id": "E0",
        "variant": "FP32",
        "checkpoint_sha256": checkpoint_hash,
        "parameter_count": parameter_count,
        "capability": {
            key: capability[key]
            for key in (
                "clean_accuracy",
                "clean_accuracy_percent",
                "macro_f1",
                "negative_log_likelihood",
                "expected_calibration_error_15_bin",
            )
        },
        "resources": resources,
        "edf_score": 1.0,
        "edf_normalization_reference": "E0",
        "capability_score": None,
        "cpedf_score": None,
        "scoring_status": (
            "pending CIFAR-10-C, failure-consistency comparison, and approved weights"
        ),
        "protocol_freeze_authorized": False,
    }

    write_json(args.output_dir / "capability_metrics.json", capability)
    write_json(args.output_dir / "resource_metrics.json", resources)
    write_csv(args.output_dir / "latency_raw.csv", raw_latency)
    write_csv(args.output_dir / "predictions.csv", predictions)
    write_json(args.output_dir / "environment.json", environment)
    write_json(args.output_dir / "edf_summary.json", summary)
    manifest = create_manifest(args.output_dir, config, started_at)
    write_json(args.output_dir / "manifest.json", manifest)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nEDF evidence written to: {args.output_dir}")


if __name__ == "__main__":
    main()

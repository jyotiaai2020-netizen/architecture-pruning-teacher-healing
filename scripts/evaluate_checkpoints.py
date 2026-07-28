from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import psutil
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10
CLASS_NAMES = (
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
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model() -> nn.Module:
    model = resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    return model


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def build_test_loader(
    root: str,
    batch_size: int,
    num_workers: int,
    download: bool,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint is not a dictionary.")

    if checkpoint and all(
        isinstance(value, torch.Tensor)
        for value in checkpoint.values()
    ):
        return checkpoint

    for key in ("state_dict", "model_state_dict", "model"):
        candidate = checkpoint.get(key)

        if isinstance(candidate, dict):
            return candidate

    raise ValueError("No compatible state_dict found in checkpoint.")


def calculate_confusion_matrix(
    labels: torch.Tensor,
    predictions: torch.Tensor,
) -> torch.Tensor:
    indices = labels * NUM_CLASSES + predictions
    counts = torch.bincount(
        indices,
        minlength=NUM_CLASSES * NUM_CLASSES,
    )
    return counts.reshape(NUM_CLASSES, NUM_CLASSES)


def calculate_macro_f1(confusion: torch.Tensor) -> float:
    confusion = confusion.float()
    true_positive = confusion.diag()
    false_positive = confusion.sum(dim=0) - true_positive
    false_negative = confusion.sum(dim=1) - true_positive

    denominator = (
        2.0 * true_positive
        + false_positive
        + false_negative
    )

    f1 = torch.where(
        denominator > 0,
        2.0 * true_positive / denominator,
        torch.zeros_like(denominator),
    )

    return float(f1.mean().item())


def calculate_ece(
    confidences: torch.Tensor,
    correct: torch.Tensor,
    bins: int = 15,
) -> float:
    boundaries = torch.linspace(0.0, 1.0, bins + 1)
    ece = torch.tensor(0.0)

    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]

        if index == 0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences > lower) & (confidences <= upper)

        if mask.any():
            bin_accuracy = correct[mask].float().mean()
            bin_confidence = confidences[mask].mean()
            ece += mask.float().mean() * torch.abs(
                bin_accuracy - bin_confidence
            )

    return float(ece.item())


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    loader: DataLoader,
    device: torch.device,
    warmup_batches: int,
) -> dict[str, Any]:
    model = build_model().to(device)

    raw_checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    state_dict = extract_state_dict(raw_checkpoint)

    incompatible = model.load_state_dict(state_dict, strict=True)

    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict checkpoint loading unexpectedly reported incompatible keys."
        )

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    process = psutil.Process(os.getpid())
    initial_ram = process.memory_info().rss / (1024**2)
    peak_ram_mb = initial_ram

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    iterator = iter(loader)

    for _ in range(warmup_batches):
        try:
            images, _ = next(iterator)
        except StopIteration:
            break

        images = images.to(device, non_blocking=True)
        _ = model(images)

    if device.type == "cuda":
        torch.cuda.synchronize()

    all_labels: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []
    all_confidences: list[torch.Tensor] = []
    all_correct: list[torch.Tensor] = []
    batch_latency_ms_per_sample: list[float] = []

    total_nll = 0.0
    total_samples = 0
    timed_seconds = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        logits = model(images)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        batch_size = labels.size(0)

        timed_seconds += elapsed
        batch_latency_ms_per_sample.append(
            1000.0 * elapsed / batch_size
        )

        probabilities = torch.softmax(logits, dim=1)
        confidences, predictions = probabilities.max(dim=1)
        correct = predictions.eq(labels)

        total_nll += criterion(logits, labels).item()
        total_samples += batch_size

        all_labels.append(labels.cpu())
        all_predictions.append(predictions.cpu())
        all_confidences.append(confidences.cpu())
        all_correct.append(correct.cpu())

        peak_ram_mb = max(
            peak_ram_mb,
            process.memory_info().rss / (1024**2),
        )

    labels_cpu = torch.cat(all_labels)
    predictions_cpu = torch.cat(all_predictions)
    confidences_cpu = torch.cat(all_confidences)
    correct_cpu = torch.cat(all_correct)

    confusion = calculate_confusion_matrix(
        labels_cpu,
        predictions_cpu,
    )

    class_totals = confusion.sum(dim=1)
    class_correct = confusion.diag()

    per_class_accuracy = {
        CLASS_NAMES[index]: (
            100.0
            * float(class_correct[index].item())
            / float(class_totals[index].item())
        )
        for index in range(NUM_CLASSES)
    }

    latency_values = np.asarray(
        batch_latency_ms_per_sample,
        dtype=np.float64,
    )

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if device.type == "cuda"
        else 0.0
    )

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_size_mib": checkpoint_path.stat().st_size
        / (1024**2),
        "strict_load": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "parameter_count": count_parameters(model),
        "sample_count": total_samples,
        "accuracy_percent": 100.0
        * float(correct_cpu.sum().item())
        / total_samples,
        "macro_f1": calculate_macro_f1(confusion),
        "nll": total_nll / total_samples,
        "ece_15_bin": calculate_ece(
            confidences_cpu,
            correct_cpu,
            bins=15,
        ),
        "median_latency_ms_per_sample": float(
            np.median(latency_values)
        ),
        "p95_latency_ms_per_sample": float(
            np.percentile(latency_values, 95)
        ),
        "throughput_samples_per_second": (
            total_samples / timed_seconds
        ),
        "timed_inference_seconds": timed_seconds,
        "peak_ram_mb": peak_ram_mb,
        "peak_vram_mb": peak_vram_mb,
        "confusion_matrix": confusion.tolist(),
        "per_class_accuracy_percent": per_class_accuracy,
    }


def log_checkpoint_metrics(
    prefix: str,
    result: dict[str, Any],
) -> None:
    mlflow.log_params(
        {
            f"{prefix}_checkpoint_path": result["checkpoint_path"],
            f"{prefix}_checkpoint_sha256": result["checkpoint_sha256"],
            f"{prefix}_strict_load": result["strict_load"],
        }
    )

    mlflow.log_metrics(
        {
            f"{prefix}_checkpoint_size_bytes":
                result["checkpoint_size_bytes"],
            f"{prefix}_checkpoint_size_mib":
                result["checkpoint_size_mib"],
            f"{prefix}_accuracy_percent":
                result["accuracy_percent"],
            f"{prefix}_macro_f1": result["macro_f1"],
            f"{prefix}_nll": result["nll"],
            f"{prefix}_ece_15_bin": result["ece_15_bin"],
            f"{prefix}_median_latency_ms_per_sample":
                result["median_latency_ms_per_sample"],
            f"{prefix}_p95_latency_ms_per_sample":
                result["p95_latency_ms_per_sample"],
            f"{prefix}_throughput_samples_per_second":
                result["throughput_samples_per_second"],
            f"{prefix}_peak_ram_mb": result["peak_ram_mb"],
            f"{prefix}_peak_vram_mb": result["peak_vram_mb"],
        }
    )

    for class_name, accuracy in result[
        "per_class_accuracy_percent"
    ].items():
        mlflow.log_metric(
            f"{prefix}_class_accuracy_{class_name}",
            accuracy,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministically evaluate CNN-002B checkpoints."
    )
    parser.add_argument("--best-checkpoint", required=True)
    parser.add_argument("--final-checkpoint", required=True)
    parser.add_argument(
        "--run-name",
        default=(
            "CNN-002B-deterministic-checkpoint-evaluation-2026-07-28"
        ),
    )
    parser.add_argument("--dataset-root", default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument(
        "--tracking-uri",
        default="http://127.0.0.1:5050",
    )
    parser.add_argument(
        "--experiment-name",
        default="CNN_Student_Baseline",
    )
    parser.add_argument(
        "--output-dir",
        default="results/CNN-002B/deterministic_evaluation",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
    )
    args = parser.parse_args()

    best_path = Path(args.best_checkpoint)
    final_path = Path(args.final_checkpoint)

    for path in (best_path, final_path):
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    loader = build_test_loader(
        root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=not args.no_download,
    )

    print(f"Device: {device}")
    print(f"Samples: {len(loader.dataset)}")
    print(f"Evaluating best checkpoint: {best_path}")

    best_result = evaluate_checkpoint(
        checkpoint_path=best_path,
        loader=loader,
        device=device,
        warmup_batches=args.warmup_batches,
    )

    print(f"Evaluating final checkpoint: {final_path}")

    final_result = evaluate_checkpoint(
        checkpoint_path=final_path,
        loader=loader,
        device=device,
        warmup_batches=args.warmup_batches,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "protocol": {
            "status": "executed_not_yet_frozen",
            "dataset": "CIFAR10",
            "dataset_split": "official_test",
            "sample_count": len(loader.dataset),
            "shuffle": False,
            "normalization_mean": CIFAR10_MEAN,
            "normalization_std": CIFAR10_STD,
            "model": "resnet18_cifar10",
            "seed": args.seed,
            "batch_size": args.batch_size,
            "warmup_batches": args.warmup_batches,
            "device": str(device),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "python_version": platform.python_version(),
            "git_commit": get_git_commit(),
        },
        "best": best_result,
        "final": final_result,
        "comparison": {
            "hashes_identical": (
                best_result["checkpoint_sha256"]
                == final_result["checkpoint_sha256"]
            ),
            "accuracy_difference_best_minus_final":
                best_result["accuracy_percent"]
                - final_result["accuracy_percent"],
            "recommended_checkpoint": (
                "best"
                if best_result["accuracy_percent"]
                >= final_result["accuracy_percent"]
                else "final"
            ),
        },
    }

    report_path = output_dir / "evaluation_report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(
            {
                "experiment_id": "CNN-002B",
                "evaluation_type":
                    "deterministic_checkpoint_comparison",
                "model": "resnet18_cifar10",
                "dataset": "CIFAR10",
                "dataset_split": "official_test",
                "sample_count": len(loader.dataset),
                "seed": args.seed,
                "batch_size": args.batch_size,
                "shuffle": False,
                "normalization_mean": str(CIFAR10_MEAN),
                "normalization_std": str(CIFAR10_STD),
                "device": str(device),
                "parameter_count":
                    best_result["parameter_count"],
                "git_commit": get_git_commit(),
                "torch_version": torch.__version__,
                "torchvision_version": torchvision.__version__,
                "recommended_checkpoint":
                    report["comparison"]["recommended_checkpoint"],
            }
        )

        log_checkpoint_metrics("best", best_result)
        log_checkpoint_metrics("final", final_result)

        mlflow.log_metric(
            "accuracy_difference_best_minus_final",
            report["comparison"][
                "accuracy_difference_best_minus_final"
            ],
        )

        mlflow.log_artifact(
            str(report_path),
            artifact_path="evaluation",
        )

    print(json.dumps(report, indent=2))
    print(f"\nReport saved to: {report_path}")
    print("MLflow evaluation run completed.")


if __name__ == "__main__":
    main()

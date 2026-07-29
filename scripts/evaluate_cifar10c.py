#!/usr/bin/env python3
"""Evaluate the canonical E0 FP32 checkpoint on CIFAR-10-C.

Uses the standard 15 common corruptions and all five severity levels. The
CIFAR-10-C arrays are evaluation-only artifacts and must not be committed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

CANONICAL_SHA256 = "d106883bd4fda76a9bd6c15ec7d5d398bc8a39d8337a9d7da11dcd817f405e36"
EXPECTED_PARAMETERS = 11_173_962
EXPECTED_CONDITION_SAMPLES = 10_000
MEAN = np.asarray((0.4914, 0.4822, 0.4465), dtype=np.float32).reshape(1, 1, 3)
STD = np.asarray((0.2470, 0.2435, 0.2616), dtype=np.float32).reshape(1, 1, 3)
STANDARD_CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/CNN-002B/best_model.pt"))
    p.add_argument("--cifar10c-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/E0/cifar10c_v1"))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fmt(seconds: float | None) -> str:
    if seconds is None:
        return "estimating"
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def progress(message: str, started: float, eta: float | None = None) -> None:
    suffix = f" | elapsed={fmt(time.perf_counter() - started)}"
    if eta is not None:
        suffix += f" | eta={fmt(eta)}"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[PROGRESS] {stamp} | {message}{suffix}", flush=True)


def build_model() -> nn.Module:
    model = models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            if isinstance(payload.get(key), Mapping):
                payload = payload[key]
                break
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint has no recognizable state dict")
    result = {}
    for key, value in payload.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = str(key)
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        result[key] = value
    return result


def load_model(path: Path) -> tuple[nn.Module, str]:
    actual = sha256_file(path)
    if actual != CANONICAL_SHA256:
        raise RuntimeError(f"Checkpoint SHA-256 mismatch: {actual}")
    model = build_model()
    model.load_state_dict(state_dict(torch.load(path, map_location="cpu", weights_only=False)), strict=True)
    count = sum(p.numel() for p in model.parameters())
    if count != EXPECTED_PARAMETERS:
        raise RuntimeError(f"Parameter-count mismatch: {count}")
    model.eval()
    return model, actual


class ArrayDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = (self.images[index].astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(image.transpose(2, 0, 1)), int(self.labels[index])


def macro_f1(pred: torch.Tensor, target: torch.Tensor) -> float:
    values = []
    for class_id in range(10):
        pc, tc = pred.eq(class_id), target.eq(class_id)
        tp = (pc & tc).sum().item()
        fp = (pc & ~tc).sum().item()
        fn = (~pc & tc).sum().item()
        d = 2 * tp + fp + fn
        values.append(0.0 if d == 0 else 2 * tp / d)
    return float(mean(values))


def ece(prob: torch.Tensor, target: torch.Tensor, bins: int = 15) -> float:
    confidence, pred = prob.max(1)
    correct = pred.eq(target)
    result = torch.zeros((), dtype=torch.float64)
    bounds = torch.linspace(0, 1, bins + 1)
    for i in range(bins):
        selected = confidence.gt(bounds[i]) & confidence.le(bounds[i + 1])
        if selected.any():
            result += selected.double().mean() * (
                correct[selected].double().mean() - confidence[selected].double().mean()
            ).abs()
    return float(result)


@torch.inference_mode()
def evaluate(model: nn.Module, images: np.ndarray, labels: np.ndarray, batch_size: int):
    loader = DataLoader(ArrayDataset(images, labels), batch_size=batch_size, shuffle=False, num_workers=0)
    logits = torch.cat([model(batch) for batch, _ in loader])
    targets = torch.from_numpy(labels.astype(np.int64))
    probabilities = logits.softmax(1)
    predictions = probabilities.argmax(1)
    metrics = {
        "sample_count": len(labels),
        "accuracy": float(predictions.eq(targets).double().mean()),
        "accuracy_percent": float(100 * predictions.eq(targets).double().mean()),
        "error_rate": float(1 - predictions.eq(targets).double().mean()),
        "macro_f1": macro_f1(predictions, targets),
        "negative_log_likelihood": float(nn.functional.cross_entropy(logits, targets)),
        "expected_calibration_error_15_bin": ece(probabilities, targets),
    }
    return metrics, predictions, probabilities.max(1).values


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    if args.batch_size < 1 or args.cpu_threads < 1:
        raise ValueError("batch size and CPU threads must be positive")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != str(args.cpu_threads):
            raise RuntimeError(f"{name} must equal {args.cpu_threads}")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    started = time.perf_counter()
    progress("CIFAR-10-C evaluation started", started)
    model, checkpoint_hash = load_model(args.checkpoint)
    root = args.cifar10c_root
    labels_path = root / "labels.npy"
    required = [labels_path] + [root / f"{name}.npy" for name in STANDARD_CORRUPTIONS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing CIFAR-10-C files: " + ", ".join(missing))
    labels = np.load(labels_path, mmap_mode="r")
    if labels.shape != (50_000,):
        raise RuntimeError(f"Unexpected labels shape: {labels.shape}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    condition_rows: list[dict[str, Any]] = []
    predictions_file = None
    writer = None
    if args.save_predictions:
        predictions_file = (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(predictions_file, fieldnames=(
            "corruption", "severity", "sample_index", "target", "prediction", "confidence", "correct"
        ))
        writer.writeheader()

    total = len(STANDARD_CORRUPTIONS) * 5
    completed = 0
    dataset_hashes = {"labels.npy": sha256_file(labels_path)}
    try:
        for corruption_index, corruption in enumerate(STANDARD_CORRUPTIONS, 1):
            path = root / f"{corruption}.npy"
            dataset_hashes[path.name] = sha256_file(path)
            images = np.load(path, mmap_mode="r")
            if images.shape != (50_000, 32, 32, 3):
                raise RuntimeError(f"Unexpected {path.name} shape: {images.shape}")
            for severity in range(1, 6):
                progress(
                    f"Condition {completed + 1}/{total}: {corruption}, severity={severity}",
                    started,
                )
                section = slice((severity - 1) * 10_000, severity * 10_000)
                metrics, predictions, confidence = evaluate(
                    model, images[section], labels[section], args.batch_size
                )
                condition_rows.append({"corruption": corruption, "severity": severity, **metrics})
                if writer is not None:
                    for index in range(EXPECTED_CONDITION_SAMPLES):
                        target = int(labels[section][index])
                        prediction = int(predictions[index])
                        writer.writerow({
                            "corruption": corruption,
                            "severity": severity,
                            "sample_index": index,
                            "target": target,
                            "prediction": prediction,
                            "confidence": float(confidence[index]),
                            "correct": prediction == target,
                        })
                    predictions_file.flush()
                completed += 1
                elapsed = time.perf_counter() - started
                progress(
                    f"Completed {completed}/{total}; accuracy={metrics['accuracy_percent']:.2f}%",
                    started,
                    elapsed / completed * (total - completed),
                )
    finally:
        if predictions_file is not None:
            predictions_file.close()

    for row in condition_rows:
        row["corruption_mean_accuracy"] = float(mean(
            r["accuracy"] for r in condition_rows if r["corruption"] == row["corruption"]
        ))
    summary = {
        "schema_version": "cifar10c_v1",
        "experiment_id": "E0",
        "variant": "FP32",
        "checkpoint_sha256": checkpoint_hash,
        "dataset": {
            "name": "CIFAR-10-C",
            "source": "https://zenodo.org/records/2535967",
            "standard_corruptions": list(STANDARD_CORRUPTIONS),
            "severity_levels": [1, 2, 3, 4, 5],
            "condition_count": total,
            "image_evaluations": total * EXPECTED_CONDITION_SAMPLES,
            "file_sha256": dataset_hashes,
        },
        "robustness": {
            "mean_corruption_accuracy": float(mean(r["accuracy"] for r in condition_rows)),
            "mean_corruption_error": float(mean(r["error_rate"] for r in condition_rows)),
            "mean_macro_f1": float(mean(r["macro_f1"] for r in condition_rows)),
            "mean_negative_log_likelihood": float(mean(r["negative_log_likelihood"] for r in condition_rows)),
            "mean_ece_15_bin": float(mean(r["expected_calibration_error_15_bin"] for r in condition_rows)),
            "aggregation": "unweighted mean across 15 corruptions and 5 severities",
            "normalized_alexnet_mce": None,
        },
        "seed": args.seed,
        "cpu_threads": args.cpu_threads,
        "batch_size": args.batch_size,
        "failure_consistency": "prediction-level reference created" if args.save_predictions else "not recorded",
        "protocol_freeze_authorized": False,
    }
    with (args.output_dir / "condition_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(condition_rows[0]))
        writer.writeheader()
        writer.writerows(condition_rows)
    write_json(args.output_dir / "robustness_summary.json", summary)
    manifest_files = {}
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            manifest_files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    write_json(args.output_dir / "manifest.json", {
        "schema_version": "cifar10c_v1",
        "status": "completed_pending_protocol_freeze",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": manifest_files,
    })
    print(json.dumps(summary, indent=2, sort_keys=True))
    progress(f"Evidence written to {args.output_dir}", started)


if __name__ == "__main__":
    main()

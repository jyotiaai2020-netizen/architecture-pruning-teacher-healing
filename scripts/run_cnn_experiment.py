from __future__ import annotations
import subprocess
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet50


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(name: str, num_classes: int) -> nn.Module:
    normalized_name = name.lower()

    if normalized_name == "tinycnn":
        return TinyCNN(num_classes=num_classes)

    if normalized_name == "resnet18":
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
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if normalized_name == "resnet50":
        model = resnet50(weights=None)
        model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def model_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024**2)


def build_loaders(
    root: str,
    batch_size: int,
    num_workers: int,
    download: bool,
) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        download=download,
        transform=train_transform,
    )

    testset = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=test_transform,
    )

    pin_memory = torch.cuda.is_available()

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return trainloader, testloader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    if device.type == "cuda":
        torch.cuda.synchronize()

    average_loss = total_loss / len(loader)
    elapsed = time.perf_counter() - start_time

    return average_loss, elapsed


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()

    correct = 0
    total = 0

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time
    accuracy = 100.0 * correct / total
    latency_ms_per_sample = 1000.0 * elapsed / total

    return accuracy, elapsed, latency_ms_per_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    experiment = config["experiment"]
    model_config = config["model"]
    dataset_config = config["dataset"]
    training = config["training"]
    tracking = config["tracking"]

    set_seed(int(experiment["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(training.get("mixed_precision", True)) and device.type == "cuda"

    print(f"Experiment: {experiment['id']}")
    print(f"Model: {model_config['name']}")
    print(f"Device: {device}")

    trainloader, testloader = build_loaders(
        root=dataset_config["root"],
        batch_size=int(training["batch_size"]),
        num_workers=int(training["num_workers"]),
        download=bool(dataset_config.get("download", True)),
    )

    model = build_model(
        name=model_config["name"],
        num_classes=int(model_config["num_classes"]),
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_name = training["optimizer"].lower()
    learning_rate = float(training["learning_rate"])

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {training['optimizer']}")

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[25, 40],
        gamma=0.1,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    checkpoint_dir = Path("checkpoints") / experiment["id"]
    result_dir = Path("results") / experiment["id"]

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking["tracking_uri"])
    mlflow.set_experiment(tracking["experiment_name"])

    process = psutil.Process(os.getpid())
    total_start = time.perf_counter()

    try:
        git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()
    except Exception:
        git_commit = "unknown"

    with mlflow.start_run(run_name=experiment["name"]):
        mlflow.log_params(
            {
                "experiment_id": experiment["id"],
                "model": model_config["name"],
                "role": model_config["role"],
                "dataset": dataset_config["name"],
                "seed": experiment["seed"],
                "epochs": training["epochs"],
                "batch_size": training["batch_size"],
                "learning_rate": training["learning_rate"],
                "optimizer": training["optimizer"],
                "device": str(device),
                "mixed_precision": use_amp,
                "parameter_count": count_parameters(model),
                "git_commit": git_commit,
            }
        )

        best_accuracy = 0.0

        for epoch in range(1, int(training["epochs"]) + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            current_learning_rate = optimizer.param_groups[0]["lr"]

            train_loss, train_seconds = train_one_epoch(
                model=model,
                loader=trainloader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                use_amp=use_amp,
            )

            accuracy, eval_seconds, latency_ms = evaluate(
                model=model,
                loader=testloader,
                device=device,
            )

            peak_ram_mb = process.memory_info().rss / (1024**2)
            peak_vram_mb = (
                torch.cuda.max_memory_allocated() / (1024**2)
                if device.type == "cuda"
                else 0.0
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "test_accuracy": accuracy,
                    "train_time_seconds": train_seconds,
                    "eval_time_seconds": eval_seconds,
                    "latency_ms_per_sample": latency_ms,
                    "peak_ram_mb": peak_ram_mb,
                    "peak_vram_mb": peak_vram_mb,
                    "learning_rate": current_learning_rate,
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch}: "
                f"loss={train_loss:.4f}, "
                f"accuracy={accuracy:.2f}%, "
                f"lr={current_learning_rate:.6f}, "
                f"train={train_seconds:.2f}s, "
                f"VRAM={peak_vram_mb:.2f} MB"
            )

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(
                    model.state_dict(),
                    checkpoint_dir / "best_model.pt",
                )

            scheduler.step()

        final_checkpoint = checkpoint_dir / "final_model.pt"
        torch.save(model.state_dict(), final_checkpoint)

        total_seconds = time.perf_counter() - total_start
        size_mb = model_size_mb(final_checkpoint)

        summary = {
            "experiment_id": experiment["id"],
            "model": model_config["name"],
            "device": str(device),
            "best_accuracy": best_accuracy,
            "parameters": count_parameters(model),
            "model_size_mb": size_mb,
            "wall_clock_seconds": total_seconds,
        }

        summary_path = result_dir / "summary.json"

        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)

        mlflow.log_metrics(
            {
                "best_accuracy": best_accuracy,
                "model_size_mb": size_mb,
                "wall_clock_seconds": total_seconds,
            }
        )

        # Log experiment outputs.
        mlflow.log_artifact(str(summary_path), artifact_path="results")
        mlflow.log_artifact(str(final_checkpoint), artifact_path="checkpoints")
        mlflow.log_artifact(args.config, artifact_path="config")

        # Log reproducibility and hardware metadata.
        environment_files = [
            "metadata/environment/nvidia-smi.txt",
            "metadata/environment/python-version.txt",
            "metadata/environment/pip-packages.txt",
            "metadata/environment/system.txt",
        ]

        for environment_file in environment_files:
            environment_path = Path(environment_file)

            if environment_path.exists():
                mlflow.log_artifact(
                    str(environment_path),
                    artifact_path="environment",
                )
            else:
                print(
                    f"Warning: environment metadata not found: "
                    f"{environment_path}"
                )

    print("Experiment completed.")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from thop import profile

from scripts.run_cnn_experiment_cnn003b_validation import (
    build_model,
    evaluate,
)

TEACHER = Path("checkpoints/CNN-003B/best_model.pt")
EXPECTED_SHA = "4d3ca6a7b1f0396e4e56024341144d487aa173ace0c7e5d981eaf9879a4ae31a"
MANIFEST = Path("evidence/CNN-003B/cifar10_train45k_val5k_seed42.json")
OUTPUT = Path("results/CNN-004")
EXPECTED_PARAMS = 20218770
EXPECTED_MACS = 1134338752.0


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert sha256(TEACHER) == EXPECTED_SHA, "Teacher checksum mismatch"

    torch.manual_seed(42)

    model = build_model("resnet50", 10)
    model.load_state_dict(
        torch.load(TEACHER, map_location="cpu", weights_only=True)
    )
    model.eval()

    example = torch.randn(1, 3, 32, 32)

    ignored = [model.fc, model.conv1]
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1):
            ignored.append(module)

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example,
        importance=tp.importance.MagnitudeImportance(p=2),
        pruning_ratio=0.20,
        iterative_steps=1,
        global_pruning=False,
        ignored_layers=ignored,
        root_module_types=[nn.Conv2d],
    )
    pruner.step()
    model.eval()

    parameters = sum(p.numel() for p in model.parameters())

    with torch.no_grad():
        macs, _ = profile(model, inputs=(example,), verbose=False)
        assert tuple(model(example).shape) == (1, 10)

    assert parameters == EXPECTED_PARAMS, (
        f"Parameter mismatch: {parameters}"
    )
    assert abs(macs - EXPECTED_MACS) < 1, (
        f"MAC mismatch: {macs}"
    )

    manifest = json.loads(MANIFEST.read_text())
    assert manifest["seed"] == 42
    val_indices = manifest["val_indices"]
    train_indices = manifest["train_indices"]

    assert len(val_indices) == 5000
    assert len(train_indices) == 45000
    assert len(set(val_indices)) == 5000
    assert not (set(val_indices) & set(train_indices))
    assert set(val_indices) | set(train_indices) == set(range(50000))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="./datasets",
        train=True,
        download=False,
        transform=transform,
    )

    loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.inference_mode():
        accuracy, elapsed, latency = evaluate(model, loader, device)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Store tensor weights only; reconstruct architecture using
    # the same versioned pruning recipe before loading.
    weights_path = OUTPUT / "cnn004_pruned_before_recovery_state_dict.pt"
    torch.save(model.cpu().state_dict(), weights_path)

    result = {
        "experiment_id": "CNN-004",
        "stage": "pruned_before_recovery",
        "teacher_sha256": EXPECTED_SHA,
        "split_manifest_sha256": sha256(MANIFEST),
        "pruning_ratio": 0.20,
        "torch_pruning_version": tp.__version__,
        "parameters": parameters,
        "macs": macs,
        "validation_samples": len(val_indices),
        "teacher_best_validation_accuracy_percent": 92.38,
        "pruned_validation_accuracy_percent": accuracy,
        "validation_accuracy_change_pp": accuracy - 92.38,
        "validation_elapsed_seconds": elapsed,
        "validation_ms_per_sample": latency,
        "weights_path": str(weights_path),
        "weights_sha256": sha256(weights_path),
    }

    result_path = OUTPUT / "cnn004_pruned_validation.json"
    if result_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing result: {result_path}"
        )

    result_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print("CNN-004 validation evaluation: COMPLETE")


if __name__ == "__main__":
    main()

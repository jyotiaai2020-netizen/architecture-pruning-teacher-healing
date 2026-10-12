import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch_pruning as tp
import torchvision
from thop import profile
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from scripts.run_cnn_experiment_cnn003b_validation import build_model

TEACHER = Path("checkpoints/CNN-003B/best_model.pt")
STUDENT = Path("results/CNN-004/cnn004_pruned_before_recovery_state_dict.pt")
MANIFEST = Path("evidence/CNN-003B/cifar10_train45k_val5k_seed42.json")

TEACHER_SHA = "4d3ca6a7b1f0396e4e56024341144d487aa173ace0c7e5d981eaf9879a4ae31a"
STUDENT_SHA = "595b761259c88e4f656a1a3102ec457d41c57551db301c39111fe4157fb89057"
EXPECTED_PARAMS = 20218770
EXPECTED_MACS = 1134338752


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert sha256(TEACHER) == TEACHER_SHA, "Teacher hash mismatch"
    assert sha256(STUDENT) == STUDENT_SHA, "Student hash mismatch"

    torch.manual_seed(42)

    model = build_model("resnet50", 10)
    model.load_state_dict(
        torch.load(TEACHER, map_location="cpu", weights_only=True),
        strict=True,
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

    frozen_weights = torch.load(
    STUDENT,
    map_location="cpu",
    weights_only=True,
)

# Remove THOP profiling buffers from the in-memory copy.
# Preserve all actual model weights and BatchNorm statistics.
    profiling_keys = [
        key for key in frozen_weights
        if key.split(".")[-1] in ("total_ops", "total_params")
    ]

    for key in profiling_keys:
        del frozen_weights[key]

    print(f"Removed {len(profiling_keys)} THOP profiling entries")

# Strict loading is still required.
    model.load_state_dict(frozen_weights, strict=True)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    assert params == EXPECTED_PARAMS, f"Parameter mismatch: {params}"

    with torch.inference_mode():
        macs, _ = profile(model, inputs=(example,), verbose=False)
        assert tuple(model(example).shape) == (1, 10)

    assert abs(macs - EXPECTED_MACS) < 1, f"MAC mismatch: {macs}"

    manifest = json.loads(MANIFEST.read_text())
    val_indices = manifest["val_indices"]
    train_indices = manifest["train_indices"]

    assert manifest["seed"] == 42
    assert len(val_indices) == 5000
    assert len(train_indices) == 45000
    assert len(set(val_indices)) == 5000
    assert not (set(val_indices) & set(train_indices))
    assert set(val_indices).isdisjoint(train_indices)
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
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total

    print("\nCNN-004 CLEAN RELOAD VERIFICATION")
    print("--------------------------------")
    print("Teacher SHA256: PASS")
    print("Student SHA256: PASS")
    print("Strict state_dict reload: PASS")
    print(f"Parameters: {params:,}")
    print(f"MACs/image: {macs:,.0f}")
    print(f"Validation samples: {total:,}")
    print(f"Validation accuracy: {accuracy:.4f}%")
    print(f"Device: {device}")

    assert total == 5000
    assert abs(accuracy - 92.18) < 0.0001, (
        f"Accuracy differs from frozen result: {accuracy:.4f}%"
    )

    print("CNN-004 CLEAN RELOAD: PASS")


if __name__ == "__main__":
    main()

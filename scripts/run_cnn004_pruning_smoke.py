import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch_pruning as tp
from thop import profile

from scripts.run_cnn_experiment_cnn003b_validation import build_model


TEACHER_PATH = Path("checkpoints/CNN-003B/best_model.pt")
EXPECTED_SHA256 = (
    "4d3ca6a7b1f0396e4e56024341144d487aa173ace0c7e5d981eaf9879a4ae31a"
)

OUTPUT_DIR = Path("checkpoints/CNN-004")
PRUNING_RATIO = 0.20


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def measure(model, example):
    model.eval()
    with torch.no_grad():
        macs, _ = profile(model, inputs=(example,), verbose=False)
        output = model(example)

    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "macs": macs,
        "output_shape": list(output.shape),
    }


def main():
    assert sha256(TEACHER_PATH) == EXPECTED_SHA256, (
        "Frozen teacher checkpoint integrity failure"
    )

    model = build_model("resnet50", 10)
    model.load_state_dict(
        torch.load(
            TEACHER_PATH,
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval()

    example = torch.randn(1, 3, 32, 32)
    before = measure(model, example)

    # Preserve residual-output widths and downsample projections.
    # Only internal bottleneck channels are eligible.
    ignored_layers = [model.fc, model.conv1]

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # Preserve 1x1 expansion/projection output interfaces.
            if module.kernel_size == (1, 1):
                ignored_layers.append(module)

    importance = tp.importance.MagnitudeImportance(p=2)

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example,
        importance=importance,
        pruning_ratio=PRUNING_RATIO,
        iterative_steps=1,
        global_pruning=False,
        ignored_layers=ignored_layers,
        root_module_types=[nn.Conv2d],
    )

    pruner.step()

    after = measure(model, example)

    assert after["output_shape"] == [1, 10]
    assert after["parameters"] < before["parameters"]
    assert after["macs"] < before["macs"]

    results = {
        "experiment_id": "CNN-004-SMOKE",
        "teacher_sha256": EXPECTED_SHA256,
        "torch_pruning_version": tp.__version__,
        "pruning_ratio_target": PRUNING_RATIO,
        "before": before,
        "after": after,
        "parameter_reduction_percent": (
            100 * (1 - after["parameters"] / before["parameters"])
        ),
        "mac_reduction_percent": (
            100 * (1 - after["macs"] / before["macs"])
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the architecture and weights together because structured
    # pruning changes layer dimensions.
    artifact_path = OUTPUT_DIR / "cnn004_pruned_smoke_model.pt"
    torch.save(model, artifact_path)

    results["artifact_path"] = str(artifact_path)
    results["artifact_sha256"] = sha256(artifact_path)

    with open(OUTPUT_DIR / "cnn004_pruning_smoke.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print("CNN-004 architecture smoke test: PASS")


if __name__ == "__main__":
    main()

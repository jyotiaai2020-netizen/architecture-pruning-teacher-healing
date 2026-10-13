"""Read-only CNN-005 preflight: reconstruct CNN-004 and validate recovery inputs.

Run: python -m scripts.prepare_cnn005
No training, checkpoint writes, or held-out test access.
"""
import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch_pruning as tp

from scripts.run_cnn_experiment_cnn003b_validation import build_model

TEACHER = Path("checkpoints/CNN-003B/best_model.pt")
STUDENT = Path("results/CNN-004/cnn004_pruned_before_recovery_state_dict.pt")
MANIFEST = Path("evidence/CNN-003B/cifar10_train45k_val5k_seed42.json")
EXPECTED = {
    TEACHER: "4d3ca6a7b1f0396e4e56024341144d487aa173ace0c7e5d981eaf9879a4ae31a",
    STUDENT: "595b761259c88e4f656a1a3102ec457d41c57551db301c39111fe4157fb89057",
    MANIFEST: "1c10c8261334ccc8cc3035b6fb3cfc7083b6aee45d3913971eaff11044e1d4bf",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_student():
    torch.manual_seed(42)
    model = build_model("resnet50", 10)
    model.load_state_dict(torch.load(TEACHER, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    example = torch.randn(1, 3, 32, 32)
    ignored = [model.fc, model.conv1]
    ignored.extend(m for m in model.modules()
                   if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1))
    pruner = tp.pruner.MagnitudePruner(
        model, example_inputs=example,
        importance=tp.importance.MagnitudeImportance(p=2),
        pruning_ratio=0.20, iterative_steps=1, global_pruning=False,
        ignored_layers=ignored, root_module_types=[nn.Conv2d],
    )
    pruner.step()
    state = torch.load(STUDENT, map_location="cpu", weights_only=True)
    profile_keys = [key for key in state if key.rsplit(".", 1)[-1] in ("total_ops", "total_params")]
    for key in profile_keys:
        del state[key]
    model.load_state_dict(state, strict=True)
    assert sum(p.numel() for p in model.parameters()) == 20218770
    return model, len(profile_keys)


def main():
    assert tp.__version__ == "1.6.0", f"Unexpected Torch-Pruning version: {tp.__version__}"
    for path, expected_hash in EXPECTED.items():
        assert path.is_file(), f"Missing frozen input: {path}"
        actual = sha256(path)
        assert actual == expected_hash, f"SHA-256 mismatch: {path}"
        print(f"SHA256 PASS: {path}")
    manifest = json.loads(MANIFEST.read_text())
    train = manifest["train_indices"]
    val = manifest["val_indices"]
    assert manifest["seed"] == 42
    assert len(train) == 45000 and len(val) == 5000
    assert len(set(train)) == 45000 and len(set(val)) == 5000
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(range(50000))
    a, removed_a = load_student()
    b, removed_b = load_student()
    assert removed_a == removed_b == 36
    state_a, state_b = a.state_dict(), b.state_dict()
    assert state_a.keys() == state_b.keys()
    assert all(torch.equal(state_a[k], state_b[k]) for k in state_a)
    print("CNN-005A/CNN-005B identical frozen initialization: PASS")
    print("CNN-004 parameters: 20,218,770")
    print("Train/validation: 45,000 / 5,000; no held-out test access")
    print("CNN-005 READ-ONLY PREFLIGHT: PASS")


if __name__ == "__main__":
    main()

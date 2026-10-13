"""CNN-006B: Export frozen CNN-005B FP32 validation references."""

import hashlib
from pathlib import Path

import numpy as np
import torch

from scripts.prepare_cnn005 import load_student
from scripts.run_cnn005_recovery import loaders

CHECKPOINT = Path("checkpoints/CNN-005B/best_model.pt")
EXPECTED_HASH = (
    "62eb8ec8a2826848067236fdd85aae10b06a24b3a2af9dbb929aeb16208f065e"
)

OUTPUT = Path(
    "results/CNN-006B/validation/cnn005b_fp32_reference.npz"
)


def main():
    assert not OUTPUT.exists(), "Refusing to overwrite existing evidence"

    actual_hash = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    assert actual_hash == EXPECTED_HASH, "Checkpoint hash mismatch"

    torch.set_num_threads(2)

    model, _ = load_student()
    state = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    _, loader = loaders("data", batch_size=32, workers=0)

    inputs = []
    labels = []
    logits = []

    with torch.inference_mode():
        for x, y in loader:
            z = model(x)

            inputs.append(x.numpy())
            labels.append(y.numpy())
            logits.append(z.numpy())

    x = np.concatenate(inputs)
    y = np.concatenate(labels)
    z = np.concatenate(logits)

    assert x.shape == (5000, 3, 32, 32)
    assert y.shape == (5000,)
    assert z.shape == (5000, 10)

    assert np.isfinite(x).all()
    assert np.isfinite(z).all()
    correct = int((z.argmax(axis=1) == y).sum())
    accuracy = correct / len(y)

    print(f"PyTorch FP32 correct: {correct}/5000")
    print(f"PyTorch FP32 accuracy: {accuracy:.4%}")

    assert correct == 4644, (
        "Accuracy differs from frozen CNN-006A baseline"
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        OUTPUT,
        inputs=x,
        labels=y,
        logits=z,
    )

    output_hash = hashlib.sha256(
        OUTPUT.read_bytes()
    ).hexdigest()

    print("Reference export: PASS")
    print(f"Output: {OUTPUT}")
    print(f"SHA256: {output_hash}")


if __name__ == "__main__":
    main()

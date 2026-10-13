"""CNN-006B: Export frozen CNN-005A/B models to ONNX."""
import hashlib
from pathlib import Path

import torch
import onnx

from scripts.prepare_cnn005 import load_student

CHECKPOINTS = {
    "CNN-005A": (
        Path("checkpoints/CNN-005A/best_model.pt"),
        "b974d77fb6cf6d7b2235ea0ec928a95a4e8085e930357a2b34392bc5a57b7639",
    ),
    "CNN-005B": (
        Path("checkpoints/CNN-005B/best_model.pt"),
        "62eb8ec8a2826848067236fdd85aae10b06a24b3a2af9dbb929aeb16208f065e",
    ),
}

OUTPUT = Path("results/CNN-006B/onnx")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for name, (checkpoint, expected_hash) in CHECKPOINTS.items():
        assert sha256(checkpoint) == expected_hash, f"Checkpoint hash mismatch: {name}"

        model, _ = load_student()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()

        output_path = OUTPUT / f"{name.lower()}_fp32.onnx"
        assert not output_path.exists(), f"Refusing to overwrite: {output_path}"

        example = torch.randn(1, 3, 32, 32)

        torch.onnx.export(
            model,
            example,
            str(output_path),
            opset_version=17,
            input_names=["input"],
            output_names=["logits"],
            dynamo=False,
        )

        exported = onnx.load(str(output_path))
        onnx.checker.check_model(exported)

        print(f"{name}: ONNX structural validation PASS")
        print(f"Output: {output_path}")
        print(f"SHA256: {sha256(output_path)}")


if __name__ == "__main__":
    main()

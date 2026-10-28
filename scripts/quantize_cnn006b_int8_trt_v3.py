"""CNN-006B: Deterministic training-only INT8 PTQ calibration."""

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import torchvision
from torchvision import transforms
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

SOURCE = Path("results/CNN-006B/onnx/cnn-005b_fp32.onnx")
OUTPUT = Path("results/CNN-006B/onnx/cnn-005b_int8_qdq_trt_v3.onnx")
MANIFEST = Path("results/CNN-006B/validation/int8_calibration_indices.json")
SPLIT = Path("evidence/CNN-003B/cifar10_train45k_val5k_seed42.json")

EXPECTED_MANIFEST_SHA = (
    "7a8cc1ca44a02c25b30b523a991f5b726fa241986c2f09f56c4547ced8e50565"
)
EXPECTED_SOURCE_SHA = (
    "fd3594367ce602603c4d05905ca7d8aa56bcb001bf11583f64d3de9c98588d3a"
)

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CIFARCalibrationReader(CalibrationDataReader):
    def __init__(self, indices):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

        self.dataset = torchvision.datasets.CIFAR10(
            root="data",
            train=True,
            download=False,
            transform=transform,
        )
        self.indices = indices
        self.position = 0

    def get_next(self):
        if self.position >= len(self.indices):
            return None

        index = self.indices[self.position]
        image, _ = self.dataset[index]
        self.position += 1

        array = image.unsqueeze(0).numpy().astype(np.float32)

        assert array.shape == (1, 3, 32, 32)
        assert np.isfinite(array).all()

        return {"input": array}

    def rewind(self):
        self.position = 0


def main():
    assert SOURCE.is_file(), "Missing FP32 ONNX source"
    assert MANIFEST.is_file(), "Missing calibration manifest"
    assert SPLIT.is_file(), "Missing frozen split"

    assert not OUTPUT.exists(), "Refusing to overwrite INT8 evidence"
    assert sha256(SOURCE) == EXPECTED_SOURCE_SHA, "Source ONNX SHA mismatch"
    assert sha256(MANIFEST) == EXPECTED_MANIFEST_SHA, "Manifest SHA mismatch"

    manifest = json.loads(MANIFEST.read_text())
    split = json.loads(SPLIT.read_text())

    indices = manifest["calibration_indices"]
    train = set(split["train_indices"])
    validation = set(split["val_indices"])

    assert manifest["experiment"] == "CNN-006B"
    assert manifest["calibration_count"] == 1024
    assert manifest["selection_seed"] == 42
    assert len(indices) == len(set(indices)) == 1024
    assert set(indices).issubset(train)
    assert not set(indices).intersection(validation)

    model = onnx.load(str(SOURCE))
    onnx.checker.check_model(model)

    assert len(model.graph.input) == 1
    assert model.graph.input[0].name == "input"

    reader = CIFARCalibrationReader(indices)

    print("Calibration integrity: PASS", flush=True)
    print("Calibration samples: 1024", flush=True)
    print("Method: MinMax", flush=True)
    print("Quantization: QDQ INT8 Conv/Gemm", flush=True)

    quantize_static(
        model_input=str(SOURCE),
        model_output=str(OUTPUT),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv", "Gemm"],
        per_channel=False,
        calibrate_method=CalibrationMethod.MinMax,
        calibration_providers=["CPUExecutionProvider"],
        extra_options={"QuantizeBias": False, "ActivationSymmetric": True, "WeightSymmetric": True},
    )

    quantized = onnx.load(str(OUTPUT))
    onnx.checker.check_model(quantized)

    ops = [node.op_type for node in quantized.graph.node]
    assert "QuantizeLinear" in ops
    assert "DequantizeLinear" in ops

    print("INT8 Q/DQ model validation: PASS")
    print("Output:", OUTPUT)
    print("Size bytes:", OUTPUT.stat().st_size)
    print("SHA256:", sha256(OUTPUT))


if __name__ == "__main__":
    main()

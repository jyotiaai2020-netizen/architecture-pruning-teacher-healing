"""CNN-006B: Validate TensorRT INT8 v3 against frozen PyTorch references."""

import hashlib
import json
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart


ENGINE = Path("results/CNN-006B/cnn-005b_int8_trt_v3.plan")
REFERENCE = Path(
    "results/CNN-006B/validation/cnn005b_fp32_reference.npz"
)
OUTPUT = Path(
    "results/CNN-006B/validation/cnn005b_tensorrt_int8_v3_validation.json"
)

EXPECTED_ENGINE_SHA = (
    "0681192dea27ad04383aa7a8949adc7b1bbd032ac44a0282d030ee8fb71b646d"
)
EXPECTED_REFERENCE_SHA = (
    "97015744b2f1750918e3a922f43f2a6fdd99637d3e9f17953661131acd2f3377"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_cuda(result):
    error = result[0]
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {error}")
    return result[1] if len(result) > 1 else None


def main():
    assert ENGINE.is_file(), "TensorRT engine not found"
    assert REFERENCE.is_file(), "Reference file not found"
    assert not OUTPUT.exists(), "Refusing to overwrite existing evidence"

    assert sha256(ENGINE) == EXPECTED_ENGINE_SHA
    assert sha256(REFERENCE) == EXPECTED_REFERENCE_SHA

    print("Artifact integrity: PASS")

    with np.load(REFERENCE, allow_pickle=False) as data:
        inputs = data["inputs"]
        labels = data["labels"]
        reference_logits = data["logits"]

    assert inputs.shape == (5000, 3, 32, 32)
    assert labels.shape == (5000,)
    assert reference_logits.shape == (5000, 10)

    assert inputs.dtype == np.float32
    assert np.isfinite(inputs).all()
    assert np.isfinite(reference_logits).all()

    reference_correct = int(
        (reference_logits.argmax(axis=1) == labels).sum()
    )
    assert reference_correct == 4644

    print("Reference data: PASS")
    print(f"PyTorch accuracy: {reference_correct}/5000")

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    engine = runtime.deserialize_cuda_engine(ENGINE.read_bytes())
    assert engine is not None, "Engine deserialization failed"

    context = engine.create_execution_context()
    assert context is not None

    assert engine.num_io_tensors == 2
    assert engine.get_tensor_mode("input") == trt.TensorIOMode.INPUT
    assert engine.get_tensor_mode("logits") == trt.TensorIOMode.OUTPUT

    assert tuple(context.get_tensor_shape("input")) == (1, 3, 32, 32)
    assert tuple(context.get_tensor_shape("logits")) == (1, 10)

    assert engine.get_tensor_dtype("input") == trt.float32
    assert engine.get_tensor_dtype("logits") == trt.float32

    print("TensorRT INT8 v3 engine: PASS")

    input_bytes = 3 * 32 * 32 * np.dtype(np.float32).itemsize
    output_bytes = 10 * np.dtype(np.float32).itemsize

    device_input = None
    device_output = None

    predictions = np.empty((5000, 10), dtype=np.float32)

    try:
        device_input = check_cuda(cudart.cudaMalloc(input_bytes))
        device_output = check_cuda(cudart.cudaMalloc(output_bytes))

        assert context.set_tensor_address(
            "input", int(device_input)
        )
        assert context.set_tensor_address(
            "logits", int(device_output)
        )

        for i in range(5000):
            sample = np.ascontiguousarray(
                inputs[i:i + 1], dtype=np.float32
            )
            output = np.empty((1, 10), dtype=np.float32)

            check_cuda(cudart.cudaMemcpy(
                device_input,
                sample.ctypes.data,
                input_bytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            ))

            success = context.execute_async_v3(0)
            assert success, f"TensorRT inference failed at image {i}"

            check_cuda(cudart.cudaDeviceSynchronize())

            check_cuda(cudart.cudaMemcpy(
                output.ctypes.data,
                device_output,
                output_bytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            ))

            predictions[i] = output[0]

            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/5000 images")

    finally:
        if device_output is not None:
            check_cuda(cudart.cudaFree(device_output))
        if device_input is not None:
            check_cuda(cudart.cudaFree(device_input))

    assert np.isfinite(predictions).all()
    print("TensorRT inference: PASS")

    trt_classes = predictions.argmax(axis=1)
    ref_classes = reference_logits.argmax(axis=1)

    trt_correct = int((trt_classes == labels).sum())
    agreement = int((trt_classes == ref_classes).sum())

    differences = np.abs(
        predictions.astype(np.float64)
        - reference_logits.astype(np.float64)
    )

    disagreement_indices = np.flatnonzero(
        trt_classes != ref_classes
    ).tolist()

    results = {
        "experiment": "CNN-006B",
        "model": "CNN-005B",
        "backend": "TensorRT",
        "precision": "INT8_QDQ_v3",
        "batch_size": 1,
        "validation_samples": 5000,
        "engine_sha256": EXPECTED_ENGINE_SHA,
        "reference_sha256": EXPECTED_REFERENCE_SHA,
        "pytorch_correct": reference_correct,
        "pytorch_accuracy": reference_correct / 5000,
        "tensorrt_correct": trt_correct,
        "tensorrt_accuracy": trt_correct / 5000,
        "prediction_agreement_count": agreement,
        "prediction_agreement_rate": agreement / 5000,
        "max_absolute_logit_difference": float(
            differences.max()
        ),
        "mean_absolute_logit_difference": float(
            differences.mean()
        ),
        "disagreement_indices": disagreement_indices,
    }


    results["accuracy_delta_percentage_points"] = (
        (trt_correct - reference_correct) / 5000 * 100
    )
    results["fp32_correct_int8_wrong"] = int(
        ((ref_classes == labels) & (trt_classes != labels)).sum()
    )
    results["fp32_wrong_int8_correct"] = int(
        ((ref_classes != labels) & (trt_classes == labels)).sum()
    )
    results["onnx_quantization_variant"] = "v3_symmetric_int8_fp32_bias"

    print(json.dumps(results, indent=2))

    with OUTPUT.open("x", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
        file.write("\n")

    print("Validation evidence saved:", OUTPUT)
    print("Evidence SHA256:", sha256(OUTPUT))


if __name__ == "__main__":
    main()
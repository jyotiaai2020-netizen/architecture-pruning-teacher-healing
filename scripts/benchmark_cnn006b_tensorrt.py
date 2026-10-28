
"""CNN-006B matched TensorRT batch-1 GPU execution benchmark."""

import hashlib
import json
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

ROOT = Path("results/CNN-006B")
OUTPUT = ROOT / "validation/tensorrt_fp32_int8_v3_latency.json"

ENGINES = {
    "FP32": (
        ROOT / "cnn-005b_fp32.plan",
        "57be29e147d43ef66400f60e70b15a219cc080ce47cc1a92fe414f6b13ed3eb6",
    ),
    "INT8_v3": (
        ROOT / "cnn-005b_int8_trt_v3.plan",
        "0681192dea27ad04383aa7a8949adc7b1bbd032ac44a0282d030ee8fb71b646d",
    ),
}

WARMUP = 100
ITERATIONS = 500
ROUNDS = 3


def check(result):
    if result[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA failure: {result[0]}")
    return result[1] if len(result) > 1 else None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benchmark(label, path, logger):
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None

    assert engine.num_io_tensors == 2
    assert tuple(context.get_tensor_shape("input")) == (1, 3, 32, 32)
    assert tuple(context.get_tensor_shape("logits")) == (1, 10)
    assert engine.get_tensor_dtype("input") == trt.float32
    assert engine.get_tensor_dtype("logits") == trt.float32

    sample = np.zeros((1, 3, 32, 32), dtype=np.float32)
    input_bytes = sample.nbytes
    output_bytes = 10 * np.dtype(np.float32).itemsize

    stream = None
    start = None
    end = None
    device_input = None
    device_output = None

    try:
        stream = check(cudart.cudaStreamCreate())
        start = check(cudart.cudaEventCreate())
        end = check(cudart.cudaEventCreate())
        device_input = check(cudart.cudaMalloc(input_bytes))
        device_output = check(cudart.cudaMalloc(output_bytes))

        assert context.set_tensor_address("input", int(device_input))
        assert context.set_tensor_address("logits", int(device_output))

        check(cudart.cudaMemcpyAsync(
            device_input,
            sample.ctypes.data,
            input_bytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream,
        ))
        check(cudart.cudaStreamSynchronize(stream))

        for _ in range(WARMUP):
            assert context.execute_async_v3(stream)
        check(cudart.cudaStreamSynchronize(stream))

        times_ms = []

        for _ in range(ITERATIONS):
            check(cudart.cudaEventRecord(start, stream))
            assert context.execute_async_v3(stream)
            check(cudart.cudaEventRecord(end, stream))
            check(cudart.cudaEventSynchronize(end))
            times_ms.append(float(check(
                cudart.cudaEventElapsedTime(start, end)
            )))

        values = np.asarray(times_ms, dtype=np.float64)
        summary = {
            "median_ms": float(np.median(values)),
            "mean_ms": float(np.mean(values)),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)),
            "min_ms": float(np.min(values)),
            "max_ms": float(np.max(values)),
        }
        print(label, summary, flush=True)
        return {"summary": summary, "measurements_ms": times_ms}

    finally:
        if stream is not None:
            check(cudart.cudaStreamSynchronize(stream))
        if device_output is not None:
            check(cudart.cudaFree(device_output))
        if device_input is not None:
            check(cudart.cudaFree(device_input))
        if end is not None:
            check(cudart.cudaEventDestroy(end))
        if start is not None:
            check(cudart.cudaEventDestroy(start))
        if stream is not None:
            check(cudart.cudaStreamDestroy(stream))


def main():
    assert not OUTPUT.exists(), "Refusing to overwrite benchmark evidence"

    for label, (path, expected) in ENGINES.items():
        assert path.is_file(), f"Missing {label} engine"
        assert digest(path) == expected, f"{label} SHA256 mismatch"

    logger = trt.Logger(trt.Logger.WARNING)
    results = {
        "experiment": "CNN-006B",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": "NVIDIA Tesla T4",
        "tensorrt_version": trt.__version__,
        "python_version": platform.python_version(),
        "batch_size": 1,
        "timing": "CUDA events around TensorRT enqueue; GPU execution only",
        "warmup_iterations_per_round": WARMUP,
        "measured_iterations_per_round": ITERATIONS,
        "rounds": [],
        "engine_sha256": {
            name: expected for name, (_, expected) in ENGINES.items()
        },
    }

    for round_index in range(ROUNDS):
        order = (
            ["FP32", "INT8_v3"]
            if round_index % 2 == 0
            else ["INT8_v3", "FP32"]
        )
        round_data = {"round": round_index + 1, "order": order}

        for name in order:
            path, _ = ENGINES[name]
            round_data[name] = benchmark(name, path, logger)

        results["rounds"].append(round_data)

    medians = {
        name: statistics.median(
            r[name]["summary"]["median_ms"]
            for r in results["rounds"]
        )
        for name in ENGINES
    }
    results["median_of_round_medians_ms"] = medians
    results["fp32_to_int8_speedup"] = (
        medians["FP32"] / medians["INT8_v3"]
    )

    with OUTPUT.open("x", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
        file.write("\n")

    print("Benchmark evidence:", OUTPUT)
    print("Evidence SHA256:", digest(OUTPUT))
    print("Median-of-medians:", medians)
    print("FP32/INT8 speedup:", results["fp32_to_int8_speedup"])


if __name__ == "__main__":
    main()

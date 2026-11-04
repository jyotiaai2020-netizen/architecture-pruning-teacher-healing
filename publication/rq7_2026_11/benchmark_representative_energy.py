#!/usr/bin/env python3
"""RQ7 confirmatory representative-CIFAR-10 TensorRT/T4 session.

RUN STATUS: prospective protocol only; the presence of this file is NOT evidence
that a new experiment was executed. Does not modify frozen CNN-006B evidence.
The measurement boundary excludes host copies; all 5000 frozen normalized input
tensors are preloaded to device memory before GPU energy/time measurement.
"""
import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pynvml
import tensorrt as trt
from cuda.bindings import runtime as cudart

ROOT = Path("results/CNN-006B")
REFERENCE = ROOT / "validation/cnn005b_fp32_reference.npz"
ENGINES = {
    "FP32": (ROOT / "cnn-005b_fp32.plan",
             "57be29e147d43ef66400f60e70b15a219cc080ce47cc1a92fe414f6b13ed3eb6"),
    "INT8_v3": (ROOT / "cnn-005b_int8_trt_v3.plan",
                "0681192dea27ad04383aa7a8949adc7b1bbd032ac44a0282d030ee8fb71b646d"),
}
REFERENCE_SHA = "97015744b2f1750918e3a922f43f2a6fdd99637d3e9f17953661131acd2f3377"

def checked(ret):
    if ret[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {ret[0]}")
    return ret[1] if len(ret) == 2 else (ret[1:] if len(ret) > 2 else None)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run_one(engine_path, expected_sha, images, labels, reference_classes, handle,
            rounds, warmups):
    assert sha(engine_path) == expected_sha, f"Engine SHA-256 mismatch: {engine_path}"
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError("Cannot deserialize engine; check T4/TensorRT compatibility")
    ctx = engine.create_execution_context()
    assert tuple(ctx.get_tensor_shape("input")) == (1, 3, 32, 32)
    assert tuple(ctx.get_tensor_shape("logits")) == (1, 10)
    assert engine.get_tensor_dtype("input") == trt.float32
    assert engine.get_tensor_dtype("logits") == trt.float32

    stream = checked(cudart.cudaStreamCreate())
    device_inputs = None
    device_output = None
    try:
        images = np.ascontiguousarray(images, dtype=np.float32)
        n = images.shape[0]
        per_sample = images[0].nbytes
        device_inputs = checked(cudart.cudaMalloc(images.nbytes))
        device_output = checked(cudart.cudaMalloc(10 * np.dtype(np.float32).itemsize))
        checked(cudart.cudaMemcpyAsync(
            device_inputs, images.ctypes.data, images.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
        checked(cudart.cudaStreamSynchronize(stream))
        assert ctx.set_tensor_address("logits", int(device_output))

        def enqueue(i):
            assert ctx.set_tensor_address("input", int(device_inputs + i * per_sample))
            if not ctx.execute_async_v3(stream):
                raise RuntimeError(f"TensorRT enqueue failed for image {i}")

        # Correctness is validated OUTSIDE the measured interval.
        predictions = np.empty(n, dtype=np.int64)
        out = np.empty((1, 10), dtype=np.float32)
        for i in range(n):
            enqueue(i)
            checked(cudart.cudaMemcpyAsync(
                out.ctypes.data, device_output, out.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
            checked(cudart.cudaStreamSynchronize(stream))
            if not np.isfinite(out).all():
                raise RuntimeError(f"Nonfinite prediction on image {i}")
            predictions[i] = int(out.argmax())
        accuracy = float(np.mean(predictions == labels))
        agreement = float(np.mean(predictions == reference_classes))
        if accuracy < 0.80:
            raise RuntimeError(f"Accuracy {accuracy:.4f} unexpectedly low; abort measurement")

        observations = []
        for r in range(rounds):
            for k in range(warmups):
                enqueue(k % n)
            checked(cudart.cudaStreamSynchronize(stream))

            idle_a = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            t_idle_a = time.perf_counter()
            time.sleep(3)
            idle_b = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            t_idle_b = time.perf_counter()
            idle_watts = ((idle_b - idle_a) / 1000) / (t_idle_b - t_idle_a)

            start_counter = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            start_time = time.perf_counter()
            for i in range(n):  # Exactly one model inference per frozen validation image.
                enqueue(i)
            checked(cudart.cudaStreamSynchronize(stream))
            stop_time = time.perf_counter()
            stop_counter = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            duration = stop_time - start_time
            joules = (stop_counter - start_counter) / 1000.0
            if joules <= 0 or duration <= 0:
                raise RuntimeError("Invalid energy or duration observation")
            observations.append({
                "round": r + 1, "inferences": n,
                "elapsed_seconds": duration,
                "gross_gpu_joules": joules,
                "gross_gpu_mj_per_image": joules * 1000 / n,
                "wall_ms_per_image": duration * 1000 / n,
                "idle_watts_preceding": idle_watts,
                "idle_adjusted_mj_per_image_exploratory":
                    (joules - idle_watts * duration) * 1000 / n,
            })
        return {"validation_accuracy": accuracy,
                "prediction_agreement_with_fp32_reference": agreement,
                "correct": int((predictions == labels).sum()),
                "predictions": predictions.tolist(), "rounds": observations}
    finally:
        if stream is not None:
            checked(cudart.cudaStreamSynchronize(stream))
        if device_output is not None:
            checked(cudart.cudaFree(device_output))
        if device_inputs is not None:
            checked(cudart.cudaFree(device_inputs))
        if stream is not None:
            checked(cudart.cudaStreamDestroy(stream))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-id", required=True,
                    help="Distinct session label, e.g. 2026-11-08T14-30Z-01")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmups", type=int, default=100)
    ap.add_argument("--out-dir", default="results/CNN-006B/confirmatory")
    args = ap.parse_args()
    assert args.rounds >= 1 and args.warmups >= 0
    assert args.session_id.replace("-", "").replace("_", "").isalnum(), "Safe session ID required"
    assert sha(REFERENCE) == REFERENCE_SHA, "Frozen validation NPZ checksum mismatch"
    with np.load(REFERENCE, allow_pickle=False) as z:
        images = np.ascontiguousarray(z["inputs"].astype(np.float32))
        labels = z["labels"].astype(np.int64)
        ref_classes = z["logits"].argmax(axis=1)
    assert images.shape == (5000, 3, 32, 32)
    assert labels.shape == (5000,)
    assert int((ref_classes == labels).sum()) == 4644
    assert np.isfinite(images).all()
    session = Path(args.out_dir) / args.session_id
    if session.exists():
        raise FileExistsError(f"Refusing to overwrite existing session: {session}")
    session.mkdir(parents=True, exist_ok=False)
    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu = str(pynvml.nvmlDeviceGetName(h))
        if "T4" not in gpu:
            raise RuntimeError(f"Expected Tesla T4, got {gpu}")
        report = {
            "experiment": "CNN-006B confirmatory representative-input energy",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "session_id": args.session_id, "gpu": gpu, "batch_size": 1,
            "software": {"python": platform.python_version(),
                         "tensorrt": trt.__version__,
                         "numpy": np.__version__},
            "input": "Frozen, normalized 5000-image CIFAR-10 validation NPZ, GPU-resident",
            "sample_count": len(images),
            "measurement_scope": "Gross NVML GPU-board energy; host transfers excluded",
            "reference_sha256": REFERENCE_SHA,
            "rounds": args.rounds, "warmups": args.warmups,
            "results": {}
        }
        # Alternate precision order on distinct sessions for order-effect control.
        order = ["FP32", "INT8_v3"] if args.session_id[-1] not in "13579" else ["INT8_v3", "FP32"]
        for name in order:
            path, digest = ENGINES[name]
            report["results"][name] = run_one(
                path, digest, images, labels, ref_classes, h, args.rounds, args.warmups)
        out = session / "results.json"
        with out.open("x", encoding="utf8") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote {out}; SHA256={sha(out)}")
        print("This is a GPU-board-only confirmatory measurement, not whole-system energy.")
    finally:
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pynvml
import tensorrt as trt
from cuda.bindings import runtime as cudart

ROOT = Path("results/CNN-006B")
OUTPUT = ROOT / "validation" / "tensorrt_fp32_int8_v3_energy.json"

ENGINES = {
    "FP32": ROOT / "cnn-005b_fp32.plan",
    "INT8_v3": ROOT / "cnn-005b_int8_trt_v3.plan",
}

ROUNDS = 3
WARMUP = 100
ITERATIONS = 5000
IDLE_SECONDS = 3.0


def cuda(result):
    if result[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def energy_mj(handle):
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)


def measure_idle(handle):
    start_e = energy_mj(handle)
    start_t = time.perf_counter()

    time.sleep(IDLE_SECONDS)

    end_t = time.perf_counter()
    end_e = energy_mj(handle)

    duration = end_t - start_t
    return (end_e - start_e) / 1000 / duration


def benchmark(engine_path, handle):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    assert engine is not None

    context = engine.create_execution_context()
    assert context is not None

    sample = np.zeros((1, 3, 32, 32), dtype=np.float32)
    output = np.empty((1, 10), dtype=np.float32)

    stream = None
    d_input = None
    d_output = None

    try:
        stream = cuda(cudart.cudaStreamCreate())
        d_input = cuda(cudart.cudaMalloc(sample.nbytes))
        d_output = cuda(cudart.cudaMalloc(output.nbytes))

        assert context.set_tensor_address("input", int(d_input))
        assert context.set_tensor_address("logits", int(d_output))

        cuda(cudart.cudaMemcpyAsync(
            d_input,
            sample.ctypes.data,
            sample.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream,
        ))
        cuda(cudart.cudaStreamSynchronize(stream))

        for _ in range(WARMUP):
            assert context.execute_async_v3(stream)

        cuda(cudart.cudaStreamSynchronize(stream))

        idle_watts = measure_idle(handle)

        start_energy = energy_mj(handle)
        start_time = time.perf_counter()

        for _ in range(ITERATIONS):
            assert context.execute_async_v3(stream)

        cuda(cudart.cudaStreamSynchronize(stream))

        end_time = time.perf_counter()
        end_energy = energy_mj(handle)

        elapsed = end_time - start_time
        gross_j = (end_energy - start_energy) / 1000
        idle_estimate_j = idle_watts * elapsed

        return {
            "iterations": ITERATIONS,
            "warmup_iterations": WARMUP,
            "duration_seconds": elapsed,
            "idle_power_watts": idle_watts,
            "gross_energy_joules": gross_j,
            "gross_energy_mj_per_inference": gross_j * 1000 / ITERATIONS,
            "idle_adjusted_energy_joules_estimate": gross_j - idle_estimate_j,
            "idle_adjusted_mj_per_inference_estimate":
                (gross_j - idle_estimate_j) * 1000 / ITERATIONS,
            "average_power_watts": gross_j / elapsed,
            "wall_time_ms_per_inference": elapsed * 1000 / ITERATIONS,
        }

    finally:
        if stream is not None:
            cuda(cudart.cudaStreamSynchronize(stream))
        if d_output is not None:
            cuda(cudart.cudaFree(d_output))
        if d_input is not None:
            cuda(cudart.cudaFree(d_input))
        if stream is not None:
            cuda(cudart.cudaStreamDestroy(stream))


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Evidence file already exists; refusing overwrite: {OUTPUT}"
        )

    pynvml.nvmlInit()

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        results = {
            "experiment": "CNN-006B",
            "measurement": "NVML cumulative GPU energy",
            "gpu": str(pynvml.nvmlDeviceGetName(handle)),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "batch_size": 1,
            "rounds": ROUNDS,
            "iterations_per_round": ITERATIONS,
            "scope": "GPU-resident TensorRT inference; excludes host transfers",
            "results": {"FP32": [], "INT8_v3": []},
        }

        for round_index in range(ROUNDS):
            order = (
                ["FP32", "INT8_v3"]
                if round_index % 2 == 0
                else ["INT8_v3", "FP32"]
            )

            for name in order:
                print(
                    f"Round {round_index + 1}/{ROUNDS}: {name}",
                    flush=True,
                )

                result = benchmark(ENGINES[name], handle)
                results["results"][name].append(result)

                print(
                    "  Gross energy:",
                    round(result["gross_energy_mj_per_inference"], 5),
                    "mJ/inference",
                )
                print(
                    "  Idle-adjusted estimate:",
                    round(
                        result["idle_adjusted_mj_per_inference_estimate"], 5
                    ),
                    "mJ/inference",
                )
                print(
                    "  Average power:",
                    round(result["average_power_watts"], 3),
                    "W",
                )

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(results, indent=2) + "\n")

        print("\nEnergy benchmark: PASS")
        print("Evidence:", OUTPUT)

    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()

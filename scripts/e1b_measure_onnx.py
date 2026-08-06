#!/usr/bin/env python3
"""Measure isolated-process ONNX CPU memory footprint for E1-B."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import psutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--sample-interval", type=float, default=0.01)
    return parser.parse_args()


def snapshot_memory(process: psutil.Process) -> dict[str, float]:
    info = process.memory_full_info()
    return {
        "rss_mb": info.rss / (1024**2),
        "uss_mb": info.uss / (1024**2),
    }


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    process = psutil.Process(os.getpid())
    baseline = snapshot_memory(process)

    session = ort.InferenceSession(str(args.model), sess_options=options, providers=["CPUExecutionProvider"])
    post_load = snapshot_memory(process)

    sample = np.random.randn(1, 3, 32, 32).astype(np.float32)
    for _ in range(args.warmup):
        session.run(["logits"], {"input": sample})

    samples: list[dict[str, float]] = []
    stop_event = threading.Event()

    def sampler() -> None:
        while not stop_event.is_set():
            samples.append(snapshot_memory(process))
            time.sleep(args.sample_interval)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()

    for _ in range(args.iterations):
        session.run(["logits"], {"input": sample})

    stop_event.set()
    thread.join(timeout=2)
    steady_state = snapshot_memory(process)

    peak = {
        "rss_mb": max(item["rss_mb"] for item in samples) if samples else steady_state["rss_mb"],
        "uss_mb": max(item["uss_mb"] for item in samples) if samples else steady_state["uss_mb"],
    }

    result = {
        "label": args.label,
        "model": str(args.model),
        "baseline": baseline,
        "post_load": post_load,
        "steady_state": steady_state,
        "peak": peak,
        "samples": samples,
        "warmup": args.warmup,
        "iterations": args.iterations,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

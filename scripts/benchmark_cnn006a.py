"""CNN-006A: frozen-model validation and T4 FP32 / AMP-FP16 inference benchmark."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from scripts.prepare_cnn005 import load_student
from scripts.run_cnn005_recovery import loaders
from scripts.run_cnn_experiment_cnn003b_validation import build_model

FROZEN = {
    "teacher": ("checkpoints/CNN-003B/best_model.pt", "4d3ca6a7b1f0396e4e56024341144d487aa173ace0c7e5d981eaf9879a4ae31a"),
    "pruned": ("results/CNN-004/cnn004_pruned_before_recovery_state_dict.pt", "595b761259c88e4f656a1a3102ec457d41c57551db301c39111fe4157fb89057"),
    "A": ("checkpoints/CNN-005A/best_model.pt", "b974d77fb6cf6d7b2235ea0ec928a95a4e8085e930357a2b34392bc5a57b7639"),
    "B": ("checkpoints/CNN-005B/best_model.pt", "62eb8ec8a2826848067236fdd85aae10b06a24b3a2af9dbb929aeb16208f065e"),
}
MANIFEST = Path("evidence/CNN-003B/cifar10_train45k_val5k_seed42.json")
MANIFEST_HASH = "1c10c8261334ccc8cc3035b6fb3cfc7083b6aee45d3913971eaff11044e1d4bf"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify():
    for name, (path, expected) in FROZEN.items():
        assert digest(path) == expected, f"Frozen checkpoint mismatch: {name}"
    assert digest(MANIFEST) == MANIFEST_HASH, "Split manifest mismatch"
    print("Frozen checkpoint and split hashes: PASS", flush=True)


def load_model(name):
    if name == "teacher":
        model = build_model("resnet50", 10)
    else:
        model, removed = load_student()
        assert removed == 36
    state = torch.load(FROZEN[name][0], map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().to("cuda")


@torch.inference_mode()
def evaluate(model, validation_loader, amp):
    correct = total = 0
    for images, targets in validation_loader:
        images = images.to("cuda", non_blocking=True)
        targets = targets.to("cuda", non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            predictions = model(images).argmax(1)
        correct += (predictions == targets).sum().item()
        total += targets.numel()
    assert total == 5000
    return {"correct": correct, "total": total, "accuracy_pct": correct * 100 / total}


@torch.inference_mode()
def benchmark(model, batch_size, amp, warmup, iterations):
    sample = torch.randn(batch_size, 3, 32, 32, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        for _ in range(warmup):
            model(sample)
        torch.cuda.synchronize()
        durations = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(sample)
            torch.cuda.synchronize()
            durations.append((time.perf_counter() - start) * 1000)
    ordered = sorted(durations)
    return {
        "batch_size": batch_size,
        "warmup": warmup,
        "iterations": iterations,
        "median_latency_ms_per_batch": statistics.median(durations),
        "p95_latency_ms_per_batch": ordered[(95 * len(ordered) + 99) // 100 - 1],
        "throughput_images_per_sec": batch_size * iterations * 1000 / sum(durations),
        "peak_allocated_gpu_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "scope": "Model-only synchronized GPU forward pass; synthetic input, no data transfer",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--validation-batch", type=int, default=128)
    p.add_argument("--timing-batches", type=int, nargs="+", default=[1, 32])
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--output", default="results/CNN-006A/fp32_ampfp16_t4.json")
    args = p.parse_args()
    assert args.warmup > 0 and args.iterations >= 20
    assert args.validation_batch > 0 and all(b > 0 for b in args.timing_batches)
    assert torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0)
    output = Path(args.output)
    assert not output.exists(), "Refusing to overwrite prior CNN-006A results"
    verify()
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _, val_loader = loaders(args.data_root, args.validation_batch, args.workers)
    report = {
        "experiment": "CNN-006A",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "split_sha256": MANIFEST_HASH,
        "checkpoint_sha256": {k: v[1] for k, v in FROZEN.items()},
        "results": {},
    }
    for name in ("teacher", "A", "B"):
        model = load_model(name)
        report["results"][name] = {}
        for label, amp in (("FP32", False), ("AMP_FP16", True)):
            result = evaluate(model, val_loader, amp)
            result["timings"] = [
                benchmark(model, b, amp, args.warmup, args.iterations)
                for b in args.timing_batches
            ]
            report["results"][name][label] = result
            print(f"{name} {label}: {result['correct']}/5000 = {result['accuracy_pct']:.2f}%", flush=True)
        del model
        torch.cuda.empty_cache()
    output.parent.mkdir(parents=True, exist_ok=True)
    assert not output.exists()
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("Saved:", output, flush=True)


if __name__ == "__main__":
    main()

# CNN-006A — Validation, Performance and Evidence Freeze Record

**Status:** Evidence documented; final freeze subject to local synchronization and sign-off.  
**Experiment:** Frozen CIFAR-10 ResNet50 teacher and two pruned/recovered students, NVIDIA Tesla T4, PyTorch FP32 vs CUDA AMP FP16.  
**Source benchmark:** `scripts/benchmark_cnn006a.py`.

## 1. Integrity and scope

All four frozen checkpoint SHA-256 checks and the original split-manifest check **passed** at benchmark start. Models were strictly loaded without retraining. The same 5,000-image validation split was used; no held-out test access occurred.

| Input | SHA-256 |
|---|---|
| CNN-003B teacher checkpoint | `4d3ca6a7b1f0396e4e56024341144d487aa173ace0c7e5d981eaf9879a4ae31a` |
| CNN-004 pruned checkpoint | `595b761259c88e4f656a1a3102ec457d41c57551db301c39111fe4157fb89057` |
| CNN-005A recovered checkpoint | `b974d77fb6cf6d7b2235ea0ec928a95a4e8085e930357a2b34392bc5a57b7639` |
| CNN-005B teacher-guided recovered checkpoint | `62eb8ec8a2826848067236fdd85aae10b06a24b3a2af9dbb929aeb16208f065e` |
| 45k/5k seed-42 split manifest | `1c10c8261334ccc8cc3035b6fb3cfc7083b6aee45d3913971eaff11044e1d4bf` |

## 2. Method

Validation used FP32 and CUDA autocast AMP FP16. Timings used synthetic 3×32×32 inputs, batches 1 and 32, 30 warmups, 100 timed forward passes and explicit CUDA synchronization. Median and p95 are batch-forward latency; throughput is images per elapsed timed forward-pass duration. Input loading/transfers, energy, and end-to-end serving were not measured. GPU memory is peak **PyTorch allocated** memory, not total device VRAM. Platform reported Tesla T4, PyTorch 2.12.1+cu130, PyTorch CUDA 13.0.

## 3. Validation accuracy

| Model | FP32 correct/5000 | FP32 | AMP FP16 correct/5000 | AMP FP16 |
|---|---:|---:|---:|---:|
| CNN-003B teacher | 4619 | 92.38% | 4619 | 92.38% |
| CNN-005A supervised | 4632 | 92.64% | 4632 | 92.64% |
| CNN-005B teacher-guided | 4644 | 92.88% | 4642 | 92.84% |

CNN-005B AMP FP16 is 0.04 percentage points lower than its FP32 result (two images). Equal aggregate accuracy does not prove identical per-example predictions.

## 4. Original performance observations

| Model | Mode | Batch | Median ms | p95 ms | Images/s | Peak allocated MiB |
|---|---|---:|---:|---:|---:|---:|
| Teacher | FP32 | 1 | 5.098 | 5.270 | 195.2 | 105.2 |
| Teacher | FP32 | 32 | 26.240 | 26.573 | 1220.4 | 202.5 |
| Teacher | AMP FP16 | 1 | 6.255 | 6.370 | 159.5 | 108.3 |
| Teacher | AMP FP16 | 32 | 13.210 | 13.327 | 2422.4 | 154.7 |
| CNN-005A | FP32 | 1 | 5.080 | 5.171 | 196.4 | 92.9 |
| CNN-005A | FP32 | 32 | 25.057 | 25.374 | 1277.5 | 190.8 |
| CNN-005A | AMP FP16 | 1 | 6.660 | 6.871 | 149.4 | 94.7 |
| CNN-005A | AMP FP16 | 32 | 14.581 | 14.683 | 2193.7 | 143.0 |
| CNN-005B | FP32 | 1 | 5.059 | 5.174 | 197.2 | 170.8 |
| CNN-005B | FP32 | 32 | 25.091 | 25.656 | 1272.7 | 269.0 |
| CNN-005B | AMP FP16 | 1 | 6.572 | 6.802 | 151.2 | 172.9 |
| CNN-005B | AMP FP16 | 32 | 14.601 | 14.726 | 2189.9 | 221.3 |

Original JSON artifact: `results/CNN-006A/fp32_ampfp16_t4.json`. **SHA-256:** `0b66f0664c21a93492195692db18e6e6c66136ff49dca5a255a20cd9c7aafbb0`. Preserve this original without editing.

## 5. Independent-process memory verification

A and B were each loaded and benchmarked in a **separate Python process**. Both have 20,218,770 parameters and 78.18 MiB allocated after model load.

| Model | Mode | Batch | Peak allocated MiB | Median ms |
|---|---|---:|---:|---:|
| CNN-005A | FP32 | 32 | 190.68 | 24.903 |
| CNN-005B | FP32 | 32 | 190.68 | 24.996 |
| CNN-005A | AMP FP16 | 32 | 142.93 | 14.483 |
| CNN-005B | AMP FP16 | 32 | 142.93 | 14.486 |

Supplementary **terminal-transcribed** evidence: `evidence/CNN-006A/isolated_memory_verification.csv`. **SHA-256:** `9f234b1c47697c5ae3a21b918f1af8aa95d38fe7e087ded13efbdfbe3b1bf6ba`.

The anomalously high original CNN-005B peak-memory values were **not reproduced** in isolated runs. Process-state or retained allocations are possible explanations, but the cause has **not been established**. Do not replace the original observations with the isolated values or claim the anomaly's root cause is proven.

## 6. Interpretation and limits

- At batch 32, AMP FP16 increased throughput by approximately 1.72× for both recovered models and reduced isolated peak allocated memory by about 25%.
- At batch 1, AMP FP16 was slower than FP32 in the original benchmark.
- CNN-005B had the highest validation accuracy under both modes; the single seed/split does not support statistical-significance claims.
- AMP FP16 autocast is **not** INT8 quantization, TensorRT, ONNX GPU inference, or permanent FP16 checkpoint conversion.
- Single-run timing, synthetic inputs, model-only measurements, and potential CUDA process-state effects limit generalization; energy, power, end-to-end latency, multi-seed uncertainty and deployment reliability remain unmeasured.

## 7. Preservation and freeze gate

Both raw artifacts were committed and pushed in commit `53d2f0a` on branch `experiment/cnn006-quantization`. This record is documentation of the observed evidence, **not a claim that an independent audit or final freeze sign-off has occurred**.

Before declaring formal freeze: synchronize this record locally; verify both raw-artifact SHA-256 checksums; inspect staged files; commit/push only intended CNN-006A documentation; keep all frozen CNN-003B/004/005 checkpoints and unrelated E1 changes untouched. Do not use `git add .`.

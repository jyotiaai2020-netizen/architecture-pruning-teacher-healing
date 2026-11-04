# RQ7 — CNN-006B publication reproduction package (PREPARATION, NOT NEW RESULTS)
Date: 2026-11-04. Branch: `experiment/cnn006-quantization`.

## Scientific scope
Single-model FP32-versus-INT8 TensorRT inference for CNN-005B (structured-pruned, teacher-guided-recovered ResNet-50-derived CIFAR-10 classifier) on NVIDIA Tesla T4. The tested intervention is PTQ precision only. ResNet-18 and teacher healing are not causal interventions in this experiment.

## Original frozen evidence
- Split `evidence/CNN-003B/cifar10_train45k_val5k_seed42.json`: 45,000 training and 5,000 validation; seed 42, disjoint.
- Calibration `results/CNN-006B/validation/int8_calibration_indices.json`: 1,024 training-only unique indices, zero overlap with validation.
- Accuracy `results/CNN-006B/validation/cnn005b_tensorrt_{fp32,int8_v3}_validation.json`.
- Latency `results/CNN-006B/validation/tensorrt_fp32_int8_v3_latency.json`.
- Gross GPU energy `results/CNN-006B/validation/tensorrt_fp32_int8_v3_energy.json`.
- Hash manifest `results/CNN-006B/ARTIFACT_MANIFEST.sha256`.
- Original study `results/CNN-006B/README.md` and `evidence/CNN-006B/EXPERIMENT_PROTOCOL.md`.

The original energy and latency benchmark scripts use a **repeated zero-filled GPU-resident input**, not representative validation images. The figures and statistics derived from original JSON must retain that label. No whole-system energy or carbon claim is supported.

## Binary release status — BLOCKED
As of 2026-11-04 the eight original binary assets are in draft GitHub release ID 426856548 (title *CNN-006B | INT8 Quantization and NVIDIA T4 Evaluation | v1.0*), not a published release with resolvable tag `cnn006b-int8-t4-v1.0`. A user-accessible, immutable release has **not** been finalized. Do not publish until the owner checks licensing, contents, checksums, and the release description. Publication of the full validation NPZ should receive a dataset-license and redistribution review.

For independent audit download the two TensorRT plans and the validation NPZ once release is published, then check SHA-256 values in `ARTIFACT_MANIFEST.sha256`. TensorRT plan portability is hardware/version dependent.

## Recompute statistics and vector figures (NO GPU REQUIRED)
Requirements: Python 3.10+, numpy, scipy, matplotlib.
```bash
python -m pip install numpy scipy matplotlib
python publication/rq7_2026_11/analyze_archived_evidence.py
```
Outputs are written to `publication/rq7_2026_11/analysis_outputs/`: paired statistical report, per-round CSV, and three SVG/PDF/PNG figures. These are analyses of original data, **not new experiments**. The accuracy CI is paired bootstrap from the four archived correctness cells, with 200,000 multinomial resamples and seed 20261104.

## Capture exact T4 environment (REQUIRES ORIGINAL HOST)
```bash
python publication/rq7_2026_11/capture_environment.py
```
Output includes software versions, git SHA, driver/CUDA inventory, and SHA-256 checksums, at `publication/rq7_2026_11/environment_manifest_t4.json`. This file has not yet been generated.

## Representative-image energy replication (REQUIRES T4 HOST AND RELEASE ASSETS)
Copy the original two TensorRT plans and the 5000-image `cnn005b_fp32_reference.npz` to their paths under `results/CNN-006B/`. Verify they match the manifest and do not overwrite prior frozen results. Ensure NVIDIA driver, TensorRT, `cuda-python`, `numpy`, `pynvml`/nvidia-ml-py compatible with the T4 plan. On a freshly initialized controlled GPU session:
```bash
python publication/rq7_2026_11/benchmark_representative_energy.py \
  --session-id 2026-11-08T14-30Z-01 --rounds 3 --warmups 100
```
Repeat in at least three **independent T4 sessions**, with distinct session IDs, a recorded power/clock/environment manifest per session, and observed workload exclusivity. Compare those sessions only after checking output correctness and the instrumented energy boundary. The script preloads the 5000 normalized CIFAR-10 images into GPU memory before timing, validates inference classes outside the interval, then executes one inference per image; no host transfer is counted in benchmark energy. GPU energy is still **gross board-level energy**, not whole-system energy. Unlike the baseline zero-filled experiment, it uses real validation inputs.

**Status: runner added but not executed; no representative-input energy result exists yet.** Do not copy original zero-input percentages to representative-input results.

## Missing for article submission
- Published immutable binary release and independent checksums.
- Actual representative-input outputs, session repetition and baseline uncertainty.
- Per-class comparison (predictions saved by new script after execution, but not available in original aggregate JSON).
- Complete original software environment on the T4.
- Author-affiliation, funding, competing-interest and AI-writing declarations.
- Journal-specific Guide for Authors final review and graphical exports/figure captions.

## Safe release checklist
1. Confirm all 8 expected assets and sizes and compare SHA-256 against manifest.
2. Review CIFAR-10 validation NPZ redistribution/licensing and any private metadata.
3. Record the release commit SHA and permissions.
4. Publish the existing *draft* with proper tag `cnn006b-int8-t4-v1.0` only after review.
5. Test anonymous access and downloads from a separate session; record hashes.
6. Archive a DOI if possible and cite it in manuscript.

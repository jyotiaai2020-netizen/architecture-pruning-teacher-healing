# CNN-006B — TensorRT FP32 vs INT8 Quantization

## Research question

RQ7: Does INT8 quantization provide measured hardware-efficiency improvements beyond model-size reduction?

## Experimental scope

- Source model: CNN-005B
- Hardware: NVIDIA Tesla T4
- Backend: TensorRT 10.16.1.11
- Precision comparison: FP32 versus symmetric INT8 Q/DQ v3 with FP32 bias
- Inference batch size: 1
- Frozen validation set: 5,000 examples
- Evaluation date: 2026-10-28
- GitHub release: cnn006b-int8-t4-v1.0

This experiment evaluates post-training quantization. It does not test post-quantization teacher-guided healing.

## Results

| Measurement | FP32 | INT8 v3 |
|---|---:|---:|
| Validation accuracy | 92.88% | 92.62% |
| Correct predictions | 4,644 / 5,000 | 4,631 / 5,000 |
| TensorRT engine bytes | 81,685,028 | 21,865,844 |
| GPU-stream latency, ms | 1.923296 | 0.663472 |
| Observed GPU allocation, MiB | 104 | 36 |
| Gross GPU energy, mJ/inference | 134.444 | 34.0846 |

Observed INT8 differences:
- Validation accuracy: -0.26 percentage points.
- TensorRT engine size: approximately 73.23% smaller.
- GPU-stream latency: approximately 65.50% lower (2.90x speedup).
- Observed GPU allocation: approximately 65.38% lower.
- Gross GPU energy per inference: approximately 74.65% lower.
- FP32/INT8 prediction agreement: 98.48%.
- Paired errors: 37 FP32-correct/INT8-wrong; 24 FP32-wrong/INT8-correct.

## Measurement boundaries

Latency uses CUDA events around TensorRT GPU execution, excluding host transfers and application overhead. The latency protocol uses three rounds, each with 100 warm-up and 500 measured iterations per precision.

Energy uses the NVML cumulative GPU energy counter across three rounds of 5,000 measured inferences per precision, with 100 warm-up iterations per round. Gross GPU energy includes idle consumption during measurement and excludes host transfers. It is not whole-system energy. Idle-adjusted estimates are exploratory.

GPU allocation figures are observed CUDA allocation snapshots, not peak VRAM measurements. TensorRT execution-context memory is a separate metric.

These results are hardware-, driver-, TensorRT-, and workload-specific. The TensorRT plan files may not be portable to another GPU or software environment.

## Evidence files

Validation and performance evidence:
- `validation/cnn005b_tensorrt_fp32_validation.json`
- `validation/cnn005b_tensorrt_int8_v3_validation.json`
- `validation/tensorrt_fp32_int8_v3_latency.json`
- `validation/tensorrt_fp32_int8_v3_energy.json`
- `validation/int8_calibration_indices.json`

Source scripts under repository `scripts/`:
- `export_cnn006b_onnx.py`
- `export_cnn006b_reference.py`
- `quantize_cnn006b_int8.py`
- `quantize_cnn006b_int8_trt_v2.py`
- `quantize_cnn006b_int8_trt_v3.py`
- `validate_cnn006b_tensorrt.py`
- `validate_cnn006b_tensorrt_int8_v3.py`
- `benchmark_cnn006b_tensorrt.py`
- `benchmark_cnn006b_energy.py`

## Artifact preservation

The GitHub release `cnn006b-int8-t4-v1.0` contains five ONNX files, two TensorRT engine files, and the frozen validation NPZ dataset.

The repository contains `ARTIFACT_MANIFEST.sha256`, which records original artifact checksums. All eight release binaries were downloaded and independently compared against local originals using SHA-256; all eight comparisons passed.

To verify locally, download the release assets and compare their SHA-256 values against the manifest. Release assets use flat filenames, while the manifest records original relative paths.

## Limitations and next experiments

- Conduct paired statistical inference and confidence-interval analysis.
- Evaluate calibration, class-level performance, and robustness.
- Measure end-to-end application latency and energy.
- Replicate measurements across independent sessions and additional hardware.
- Test post-quantization teacher healing as a separate controlled intervention.
- Do not report a complete Effective Deployment Footprint or Capability Efficiency score until all required inputs and normalization rules are validated.

## Research status

Evidence-preserved CNN-006B quantization checkpoint. Binary integrity verification passed. This is not a formal publication-level validation or a claim of teacher-healing efficacy.

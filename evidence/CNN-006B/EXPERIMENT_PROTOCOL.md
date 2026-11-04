# CNN-006B — Experimental Protocol

## 1. Research question

RQ7: Does INT8 quantization provide measured hardware-efficiency improvements beyond model-size reduction?

## 2. Experimental objective

Evaluate the accuracy, model size, inference latency, GPU memory allocation, and GPU energy consumption of a CNN-005B-derived model under FP32 and INT8 inference.

This experiment evaluates post-training quantization. It does not evaluate post-quantization teacher-guided healing.

## 3. Experimental environment

- Source model: CNN-005B
- GPU: NVIDIA Tesla T4
- Inference backend: TensorRT 10.16.1.11
- Precision configurations: FP32 and symmetric INT8 Q/DQ v3
- Bias precision: FP32
- Inference batch size: 1
- Frozen validation set: 5,000 examples
- Evaluation date: 2026-10-28

## 4. Experimental variables

Independent variable:
- Inference precision: FP32 versus INT8 v3

Dependent variables:
- Validation accuracy
- Model and TensorRT engine size
- GPU-stream inference latency
- Observed GPU memory allocation
- Gross GPU energy per inference

Controlled conditions:
- Source model architecture and weights
- Validation dataset
- GPU device
- TensorRT inference environment
- Batch size

## 5. Accuracy evaluation

Evaluate both TensorRT engines against the same frozen validation set.

Record:
- Correct predictions and validation accuracy
- Prediction agreement
- Paired changes in classification correctness

## 6. Latency measurement

Use CUDA events around TensorRT GPU execution.

For each precision:
- 3 measurement rounds
- 100 warm-up iterations per round
- 500 measured iterations per round
- Batch size: 1

Report the median of the three round-level median latencies.

Host transfers and application-level processing are excluded.

## 7. GPU energy measurement

Use the NVML cumulative GPU energy counter.

For each precision:
- 3 measurement rounds
- 100 warm-up iterations per round
- 5,000 measured inferences per round

Report gross GPU energy per inference.

Gross GPU energy includes idle consumption during the measurement interval. It excludes host transfers and is not whole-system energy.

Idle-adjusted estimates are exploratory.

## 8. Memory measurement

Record observed CUDA allocation snapshots after engine, execution context, and inference buffers are initialized.

These measurements are not transient peak VRAM measurements.

Keep TensorRT execution-context memory separate from observed CUDA allocations.

## 9. Evidence and traceability

Primary evidence is stored under:

- results/CNN-006B/README.md
- results/CNN-006B/validation/
- results/CNN-006B/ARTIFACT_MANIFEST.sha256

Implementation scripts are stored under scripts/.

Large binary artifacts are preserved in the GitHub draft release.

## 10. Limitations

- Results apply to the measured NVIDIA Tesla T4 configuration.
- GPU-stream latency is not end-to-end application latency.
- GPU energy is not whole-system energy.
- Observed memory allocation is not peak VRAM.
- Multi-device and independent-session replication remain necessary.
- Paired statistical inference and confidence intervals remain pending.
- Teacher-guided healing efficacy has not been tested in this experiment.

## 11. Research governance status

Status: EVIDENCE DOCUMENTATION IN PROGRESS

The protocol documents the completed measurement procedures retrospectively.

It is not evidence of prospective preregistration, independent replication, or formal research approval.

No final experiment freeze approval is asserted.

# RQ7 — Beyond Model Size: Measuring INT8 Quantization Efficiency on an NVIDIA Tesla T4

**Working research manuscript — version 0.1, 28 October 2026**  
**Status:** Pre-submission draft; no peer review, formal freeze, or independent replication.  
**Research question:** Does INT8 quantization provide measured hardware-efficiency improvements beyond model-size reduction?

## Abstract

Model compression is frequently assessed through parameter counts or file sizes, but these proxies do not directly establish runtime efficiency. This study compares FP32 and symmetric INT8 quantize/dequantize (Q/DQ) TensorRT inference for a CNN-005B-derived classifier on an NVIDIA Tesla T4 GPU. Both variants were evaluated on the same 5,000-example validation set, and GPU-stream latency, observed CUDA allocation, and cumulative GPU energy were measured under explicitly bounded protocols. INT8 validation accuracy was 92.62% versus 92.88% for FP32 (−0.26 percentage points). TensorRT engine size declined from 81,685,028 to 21,865,844 bytes (73.23% reduction). Median-of-round-median GPU-stream inference latency declined from 1.923296 to 0.663472 ms (2.90× speedup), observed CUDA allocation from 104 to 36 MiB (65.38% reduction), and median gross GPU energy per inference from 134.444 to 34.0846 mJ (74.65% reduction). These observations show efficiency improvements beyond artifact size in the measured T4 configuration. They do not establish end-to-end application performance, whole-system energy savings, cross-device generalization, statistical equivalence of predictive quality, or a causal contribution from teacher-guided healing. Independent replication and paired uncertainty analyses are required before stronger claims.

**Keywords:** post-training quantization; INT8; TensorRT; edge AI; NVIDIA T4; inference latency; GPU energy; reproducibility.

## 1. Introduction

Efficient deployment requires more than a smaller serialized model. A compressed neural network may retain high predictive accuracy yet fail to reduce device memory, execution latency, or energy on a target accelerator. Conversely, reduced precision may improve these operational measures while changing predictions. An empirical deployment assessment therefore needs to measure both predictive capability and runtime resource demand under a stated hardware and software environment.

This article addresses **RQ7** using a controlled, within-device FP32-versus-INT8 comparison. The focus is *measured* hardware efficiency beyond model size, not a claim that quantization always improves every device or workload. The work forms one component of a larger architecture-pruning and teacher-healing research program; however, the treatment isolated here is **post-training quantization**, not teacher-guided recovery.

The contributions of this working report are (i) a traceable FP32/INT8 comparison on a Tesla T4, (ii) separately bounded measurements of engine size, GPU execution latency, observed memory allocation, and gross GPU energy, and (iii) an explicit account of the inference and reproducibility limitations.

## 2. Research question and hypotheses

**RQ7:** Does INT8 quantization provide measured hardware-efficiency improvements beyond model-size reduction?

- **H1 (descriptive):** INT8 reduces serialized TensorRT engine size relative to FP32.
- **H2 (descriptive):** INT8 reduces GPU-stream inference latency in the tested T4 configuration.
- **H3 (descriptive):** INT8 reduces observed CUDA allocation in the measured configuration.
- **H4 (descriptive):** INT8 reduces gross GPU energy per inference in the measured configuration.
- **Capability constraint:** Report predictive-quality changes alongside efficiency gains; do not presume equivalence without a prespecified margin and suitable paired inference.

These hypotheses describe comparisons examined retrospectively. They were **not prospectively preregistered**.

## 3. Methods

### 3.1 Model, dataset, and deployment variants

The source is a CNN-005B-derived classifier. The comparison uses FP32 TensorRT and symmetric INT8 Q/DQ v3 with FP32 bias. Earlier quantization variants were retained as development artifacts; the validated INT8 comparator is **v3**. Both engines use batch size one. The paired accuracy comparison uses a frozen 5,000-example validation set. The calibration record reports 1,024 training-derived indices selected with seed 42 and zero validation overlap.

The model's CNN-005B lineage does **not** isolate the effect of teacher-guided healing; this study compares precision variants of the derived model.

### 3.2 Execution environment

- Accelerator: NVIDIA Tesla T4
- TensorRT: 10.16.1.11
- Driver: 595.71.05 (recorded in prior experiment environment notes)
- CUDA: 13.2 (recorded in prior experiment environment notes)
- Python: 3.12.3
- Inference batch size: 1

The complete package/build manifest and clean-environment rebuild verification are pending. TensorRT engine plans may be dependent on hardware and software versions.

### 3.3 Predictive evaluation

FP32 and INT8 predictions were compared on the same 5,000 validation examples. Recorded outputs include correct classifications, prediction agreement, and the number of cases where correctness changed between precision variants. These are **validation** results, not a separately certified final held-out test.

### 3.4 GPU-stream latency

CUDA events time TensorRT enqueue/execution only. Each precision uses three rounds, each comprising 100 warmups and 500 measured iterations. The primary reported value is the **median of three round-level median latencies**. Host transfers, preprocessing, postprocessing, and application orchestration are outside the measurement boundary.

### 3.5 Memory

Memory is reported as observed CUDA allocation snapshots after engine/context/buffer initialization and warmup. This is **not peak VRAM** and must not be conflated with separately reported TensorRT execution-context device memory.

### 3.6 GPU energy

NVML cumulative GPU energy was measured over three rounds per precision, each with 100 warmups and 5,000 measured inferences. The reported primary metric is the median gross GPU energy per inference across rounds. Gross energy includes idle power over the interval; host transfers and non-GPU system components are excluded. Idle-adjusted energy is exploratory and is not substituted for directly observed gross energy.

### 3.7 Statistical treatment

This draft reports observed differences only. McNemar testing, paired confidence intervals for accuracy change, a predeclared equivalence margin, and independent-session uncertainty analysis are **not yet completed**. Per-inference observations collected within one device/session should not be treated as independent device-level replications.

## 4. Results

### 4.1 Main comparison

| Metric | FP32 | INT8 v3 | Observed contrast |
|---|---:|---:|---|
| Validation correct / 5,000 | 4,644 | 4,631 | 13 fewer INT8 correct |
| Validation accuracy | 92.88% | 92.62% | −0.26 percentage points |
| TensorRT engine size | 81,685,028 bytes | 21,865,844 bytes | 73.23% smaller |
| GPU-stream latency | 1.923296 ms | 0.663472 ms | 65.50% lower; 2.90× speedup |
| Observed CUDA allocation | 104 MiB | 36 MiB | 65.38% lower |
| Gross GPU energy/inference | 134.444 mJ | 34.0846 mJ | 74.65% lower |

The FP32 and INT8 prediction agreement rate was 98.48% (4,924/5,000). In paired correctness comparisons, 37 examples changed from FP32-correct to INT8-wrong, while 24 changed from FP32-wrong to INT8-correct. These counts explain the net 13-example accuracy difference; they are not themselves a completed significance test.

### 4.2 Model and engine footprint

The FP32 ONNX file was 80,796,158 bytes and the INT8 v3 ONNX file was 20,388,497 bytes, approximately 74.77% smaller. The corresponding TensorRT engine reduction was approximately 73.23%. These are distinct artifact boundaries and should not be confused with installed application size or runtime memory.

### 4.3 GPU memory boundary

The observed CUDA allocation snapshots were 104 MiB for FP32 and 36 MiB for INT8. Separately, the TensorRT execution-context device-memory values were 2,725,888 and 819,200 bytes, respectively. Neither figure establishes peak device memory during all deployment conditions.

### 4.4 Energy boundary

The three recorded FP32 gross energy measurements were 133.9334, 134.4440, and 135.3450 mJ/inference. The corresponding INT8 measurements were 33.2474, 34.0846, and 34.7192 mJ/inference. The central values reported above are medians across rounds. The comparison is **GPU-only gross energy**, not total electricity consumption or battery-life improvement.

## 5. Discussion

The measured T4 configuration exhibited substantial improvements in execution latency, observed allocation, and gross GPU energy in addition to smaller engine files. This supports a narrow but useful answer to RQ7: **under the documented hardware and inference boundary, INT8 provided measured efficiency gains beyond storage reduction**.

The predictive trade-off is also material. INT8 reduced validation accuracy by 0.26 percentage points, and the two variants did not make identical predictions. Without paired uncertainty, a prespecified acceptable degradation margin, class-level analysis, calibration checks, and robustness testing, it would be premature to label predictive capability equivalent or preserved across deployment conditions.

The observed energy reduction is partly coupled to the shorter inference interval, while gross NVML energy also includes baseline power. It must not be interpreted as an independent proof of proportional system-level energy or operating-cost reduction. Nor do these T4 measurements establish performance on CPUs, phones, embedded devices, other GPUs, different batch sizes, or real application request paths.

### 5.1 Relation to the broader research program

RQ1 addresses deployable size; RQ2 addresses predictive accuracy; RQ3 and RQ4 address quality and robustness; RQ5 addresses wider deployment efficiency; and RQ6 asks whether teacher-guided healing causally improves capability. RQ7 adds a concrete hardware-measurement question on a specified accelerator. It is related to, but does not replace, those questions. A final Effective Deployment Footprint or Capability Efficiency score has not been computed.

## 6. Threats to validity

**Internal validity:** Benchmark order, GPU thermal/power behavior, clock state, and session conditions may influence observed performance. The records establish within-setup comparisons, not independent-session causal isolation of every hardware factor.

**Construct validity:** GPU-stream latency excludes end-to-end work; CUDA allocation snapshots exclude peaks; gross NVML GPU energy excludes host/system consumption. Accuracy is based on a validation set.

**Statistical conclusion validity:** Only three round-level repetitions were performed for latency and energy, and formal confidence intervals or inferential testing are pending.

**External validity:** One Tesla T4 and TensorRT configuration cannot support cross-device or production-workload generalization.

**Reproducibility:** Source scripts, JSON records, and checksums are preserved; a clean-environment rebuild and independently repeated benchmark remain to be documented.

## 7. Conclusion

For a CNN-005B-derived classifier evaluated on an NVIDIA Tesla T4, INT8 v3 TensorRT inference produced a smaller engine and lower measured GPU-stream latency, observed allocation, and gross GPU energy than FP32, accompanied by a 0.26-percentage-point decrease in validation accuracy. The evidence supports a **hardware-specific descriptive efficiency finding**, not general device efficiency, whole-system energy savings, model-quality equivalence, or teacher-healing causality. The next stage is reproducibility, paired uncertainty analysis, robustness testing, and cross-environment validation.

## 8. Reproducibility and data availability

- [Experimental protocol](../evidence/CNN-006B/EXPERIMENT_PROTOCOL.md)
- [Results summary](../results/CNN-006B/README.md)
- [Validation and benchmark JSON](../results/CNN-006B/validation/)
- [Artifact SHA-256 manifest](../results/CNN-006B/ARTIFACT_MANIFEST.sha256)
- [Research decision log](../RESEARCH_DECISION_LOG.md)
- [Research questions](../RESEARCH_QUESTIONS.md)
- [Evidence guide](../evidence/CNN-006B/README.md)

The source repository records a draft GitHub release containing eight binary artifacts; the release has not been confirmed published. Download/hash verification was recorded previously. Access and permanent archival policy should be resolved before submission.

## 9. Required work before submission

1. Reconstruct and verify a complete environment manifest, exact source commit, model provenance and run configurations.
2. Recompute all tables and uncertainty intervals from source records using a versioned analysis script.
3. Conduct McNemar and paired bootstrap analyses; define an accuracy-equivalence margin before any confirmatory equivalence claim.
4. Report per-class performance, calibration, confusion matrices, and corruption/shift robustness.
5. Repeat latency, allocation and energy benchmarks across independent sessions, recording power/clock/temperature and benchmark order.
6. Evaluate end-to-end application inference and, if relevant, whole-system energy and additional hardware.
7. Add verified, appropriate scholarly literature and references; **no citations are fabricated in this draft**.
8. Complete author contributions, funding, conflicts, data availability, ethics/applicability, target-venue formatting and peer review.

**Editorial note:** This document is a structured publishable-article *working draft*, not a claim of a finished or accepted paper.

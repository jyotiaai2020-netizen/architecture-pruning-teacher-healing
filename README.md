# Teacher-Healed Quantization for Capability-Preserving Efficient AI

This repository contains a controlled, evidence-first research program investigating whether neural-network compression—especially **static INT8 quantization**, architecture simplification, knowledge distillation, and **Teacher Healing**—can reduce deployment cost while preserving model capability.

The work is motivated by a practical question: **Can advanced AI models become smaller and easier to deploy on resource-constrained systems without losing the accuracy, quality, robustness, and decision value that make them useful?**

## Research purpose

The study evaluates compact models across two dimensions:

1. **Capability preservation** — accuracy, classification precision, recall, macro-F1, per-class behavior, calibration, prediction agreement, and robustness.
2. **Deployment efficiency** — complete deployable artifact size, latency, throughput, memory, and energy when direct measurement is available.

The intended outcome is not simply the smallest model. The preferred candidate must first satisfy declared capability and robustness gates and then demonstrate a lower **Effective Deployment Footprint (EDF)**.

```text
Capability Efficiency = Capability Score / EDF Score
```

EDF and Capability Efficiency are calculated only when every required dimension has valid evidence and the normalization policy has been declared in advance. Missing energy data is not silently replaced with an estimated value.

## Research questions

| ID | Research question | Current interpretation |
|---|---|---|
| RQ1 | Can INT8 quantization materially reduce the complete deployable model size? | **Supported for E1.** INT8 is 74.83% smaller than FP32 when the FP32 external-data file is included. |
| RQ2 | Can quantization preserve clean-test accuracy? | **Provisionally supported.** Observed CIFAR-10 accuracy is 93.81% for FP32 and 93.82% for INT8; paired significance and equivalence analyses remain pending. |
| RQ3 | Are classification precision and overall predictive quality preserved? | **Open.** Per-class precision/recall/F1, NLL, ECE, Brier score, prediction agreement, and paired bootstrap evidence are not yet final. |
| RQ4 | Does the compressed model preserve robustness under distribution shift and corruption? | **Open.** CIFAR-10-C evaluation across 15 corruptions and 5 severity levels remains pending. |
| RQ5 | Does compression improve real deployment efficiency? | **Open.** Controlled CPU latency, throughput, independent-process RSS/USS memory, and energy disposition remain pending. |
| RQ6 | Does Teacher Healing causally improve capability preservation beyond ordinary quantization or distillation? | **Not yet tested by E1.** E1 establishes the FP32-versus-INT8 validation baseline; later controlled ablations must isolate the Teacher Healing contribution. |

## Research design

The program follows a staged comparative design:

| Stage | Purpose |
|---|---|
| Baseline establishment | Train and preserve reproducible teacher and student baselines with dataset, configuration, environment, and artifact lineage. |
| Controlled compression | Apply architecture simplification, pruning, distillation, quantization, and Teacher Healing as separately traceable interventions. |
| Clean capability evaluation | Compare models on identical test observations using paired predictions rather than aggregate accuracy alone. |
| Quality and calibration | Measure class-level precision, recall, F1, confusion matrices, NLL, ECE, Brier score, and model agreement. |
| Robustness evaluation | Evaluate CIFAR-10-C using the same corruption files, corruption order, severity levels, and preprocessing for each candidate. |
| Deployment evaluation | Measure complete artifact size, controlled latency and throughput, fresh-process memory, and direct energy only when a valid counter exists. |
| Statistical validation | Use McNemar analysis and paired bootstrap confidence intervals with predeclared gates and seeds. |
| Governance and freeze | Generate immutable evidence, hashes, conformance records, and a formal decision before freezing an experiment. |

## Experiment roadmap

| Experiment | Model or intervention | Purpose | Status |
|---|---|---|---|
| CNN-001 | TinyCNN | Framework and pipeline validation | Completed |
| CNN-002 / CNN-002B | ResNet-18 student baseline | Establish reproducible clean-capability baseline | Completed |
| CNN-003 | ResNet-50 teacher | Establish higher-capability teacher baseline | Planned / not frozen |
| CNN-004 | Simplified or structurally pruned student | Measure architecture-reduction effects | Planned |
| CNN-005 | Simplified student + knowledge distillation | Isolate distillation benefit | Planned |
| CNN-006 | Distilled student + quantization | Measure combined compression effects | Planned |
| CNN-007 | Full Teacher-Healing intervention | Test recovery of capability after compression | Planned |
| E1 | ResNet-18 FP32 vs static INT8 QDQ | Validate quantization, evidence controls, and EDF readiness | Closure in progress |

## E1 current results

Status as of **August 7, 2026**:

| Measure | FP32 | Static INT8 QDQ | Interpretation |
|---|---:|---:|---|
| CIFAR-10 correct predictions | 9,381 / 10,000 | 9,382 / 10,000 | One additional correct prediction for INT8; not evidence of superiority |
| Clean-test accuracy | 93.81% | 93.82% | Observed difference: +0.01 percentage points |
| Complete deployable size | 42.718 MiB | 10.753 MiB | Includes the FP32 ONNX external-data file |
| Compression ratio | — | 3.973× smaller | Verified artifact boundary |
| Storage reduction | — | 74.83% | Verified |
| Class-level precision/recall/F1 | Pending | Pending | Required for quality-preservation conclusion |
| Calibration and paired statistics | Pending | Pending | McNemar, bootstrap, NLL, ECE, and Brier required |
| CIFAR-10-C robustness | Pending | Pending | Required before capability preservation is confirmed |
| CPU latency and throughput | Pending | Pending | Must use identical controlled settings |
| Independent-process RSS/USS memory | Pending | Pending | Each model must run in a fresh subprocess |
| Direct energy | Availability check pending | Availability check pending | Record as unavailable if no valid hardware counter exists |
| EDF / Capability Efficiency | Incomplete | Incomplete | Not calculated until all governed inputs are eligible |

### Current E1 conclusion

Static INT8 quantization has **verified a 74.83% reduction in complete deployable size** while the observed aggregate clean-test accuracy remains essentially unchanged. This supports continued evaluation, but it does **not yet prove** preservation of classification precision, calibration, class-level quality, robustness, or full deployment efficiency. E1 is therefore **preserved and in closure validation, but not formally frozen**.

## E1 validation status

| Validation item | Status |
|---|---|
| Project environment recreated and recorded | Complete |
| ONNX 1.17.0 and ONNX Runtime 1.20.1 recorded | Complete |
| CPUExecutionProvider availability verified | Complete |
| FP32 external-data reference inspected | Complete |
| All referenced model files present | Complete |
| Complete FP32 and INT8 artifact boundaries measured | Complete |
| Environment and model preflight hashes preserved | Complete |
| Remote Git checkpoint verified | Complete |
| Paired predictions and statistical analysis | Pending |
| Calibration and per-class quality analysis | Pending |
| CIFAR-10-C robustness matrix | Pending |
| Controlled latency, throughput, and memory runs | Pending |
| Energy availability and governance disposition | Pending |
| Corrected closure summary and final checksum manifest | Pending |
| Governance approval, PR merge, and formal freeze record | Pending |

## Controlled E1 performance protocol

The FP32 and INT8 models must be benchmarked under identical conditions:

- ONNX Runtime `CPUExecutionProvider`
- Batch size 1
- One intra-op thread and one inter-op thread
- Sequential execution
- 50 warm-up iterations
- 500 measured iterations per repetition
- At least 5 independent repetitions
- Identical input order, CPU affinity, process image, and instance
- Median as the primary latency statistic, with mean, SD, p90, p95, and p99
- Separate fresh subprocesses for each model's RSS and USS measurements

Performance observations collected across different EC2 restarts are not combined into one repetition set.

## Evidence and reproducibility principles

- Every result must link to the exact dataset, manifest, model artifact, configuration, source commit, environment, and SHA-256 hash.
- External ONNX data files are part of the deployable boundary.
- Capability comparisons use paired predictions on identical observations.
- Historical evidence is corrected through a new closure record rather than edited silently.
- Energy is reported as `UNAVAILABLE` when no reliable hardware counter exists.
- An experiment is frozen only after all mandatory gates pass and the final checksum manifest validates.

## Environment

- AWS EC2 `g4dn.xlarge`
- NVIDIA Tesla T4, 16 GB VRAM
- Ubuntu 24.04
- Python virtual environment
- PyTorch
- ONNX 1.17.0
- ONNX Runtime 1.20.1
- MLflow
- CIFAR-10 and CIFAR-10-C

## Current project status

**E1 closure is in progress.** The model artifacts, environment record, external-data boundary, deployable-size result, and remote checkpoint are preserved. The next work session should complete paired capability and quality analysis, CIFAR-10-C robustness, and controlled performance/memory measurement before EDF eligibility and formal freeze are decided.

# Teacher-Healed Quantization for Capability-Preserving Efficient AI

A reproducible research framework for evaluating whether architecture-aware simplification, structured pruning, knowledge distillation, quantization, and teacher-guided recovery can produce compact neural networks with acceptable capability and a lower Effective Deployment Footprint (EDF).

## Research objective

Evaluate model capability alongside parameter count, multiply-accumulate operations (MACs), checkpoint size, inference latency, memory, energy (where measured), and deployment constraints. The intended decision criterion is **Capability Efficiency = Capability Score / EDF Score**, subject to minimum capability requirements. A complete EDF evaluation has **not** yet been performed.

## Experiment status (updated 13 October 2026)

| Experiment | Purpose | Status |
|---|---|---|
| CNN-001 | TinyCNN pipeline validation | Completed historical experiment |
| CNN-002 / CNN-002B | ResNet-18 student baselines | Completed historical experiments |
| CNN-003 | Earlier ResNet-50 teacher protocol | Historical; superseded for pruning comparisons |
| **CNN-003B** | Validation-corrected CIFAR-10 ResNet-50 teacher | **Frozen single-run teacher baseline** |
| **CNN-004** | Structured channel pruning before recovery | **Frozen validated candidate; checkpoint and validation JSON archived; clean reload passed** |
| **CNN-005A** | Supervised recovery from CNN-004 | **Completed, verified, research-frozen — 92.64% validation** |
| **CNN-005B** | Teacher-guided recovery from the same CNN-004 weights | **Completed, verified, research-frozen — 92.88% validation** |
| CNN-006 | Quantization/compression ablations | Planned |
| CNN-007 | Combined method and edge deployment evaluation | Planned |
| E1 | ResNet-18 FP32 vs static INT8 QDQ | Separate study; closure validation in progress |

CNN-005A/B are completed recovery experiments; CNN-006/007 are the next planned experiments. A frozen single-run baseline is **not** equivalent to statistical replication or publication-level validation.

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

## CNN-003B completed teacher baseline — October 12, 2026

The validation-corrected CIFAR-10 ResNet-50 teacher run completed **50 epochs** on an NVIDIA Tesla T4 GPU. The official training data was deterministically divided into **45,000 training** and **5,000 validation** examples using split seed **42**, with **zero overlap**. Validation accuracy was used for best-checkpoint selection; final test accuracy was reported separately.

| Metric | Observed result |
|---|---:|
| Best validation accuracy | **92.38%** |
| Best checkpoint epoch | **48** |
| Final reported test accuracy | **92.11%** |
| Validation–test gap | 0.27 percentage points |
| Model parameters | 23,520,842 |
| Reported model size | 90.03 MiB |
| Total experiment time | 2,662.68 seconds (44 min 23 s) |
| Model | CIFAR-adapted ResNet-50 |
| Selection metric | `val_accuracy` |
| MLflow experiment | `CNN_Teacher_Baseline` |
| MLflow run ID | `edc07a6c32b0485db5db7e53cff30082` |

The EC2 run produced two checkpoint files, approximately 91 MB each:

```text
4d3ca6a7b1f0396e4e56024341144d487aa173ace0c7e5d981eaf9879a4ae31a  checkpoints/CNN-003B/best_model.pt
e63a3de3dee4c2e27c224b0da52883389b10749e799356ef380480ec5b403d9b  checkpoints/CNN-003B/final_model.pt
```

### Reproducibility and artifacts

- Runner: `scripts/run_cnn_experiment_cnn003b_validation.py`
- Configuration: `configs/cnn/cnn003b_validation_50ep.yaml`
- Split manifest: `metadata/splits/cifar10_train45k_val5k_seed42.json`
- Results: `results/CNN-003B/summary.json`
- Evidence: `evidence/CNN-003B/`
- Training log: `cnn003b_teacher_50ep.log` (also copied into evidence on EC2)
- Checkpoint fingerprints: `evidence/CNN-003B/checkpoint_sha256.txt`

These paths describe artifacts observed or prepared on EC2. Their availability in a particular GitHub branch depends on whether that branch's research commit has been pushed. The large checkpoint binaries are **not claimed to be hosted on GitHub**.

**Research interpretation:** CNN-003B is a **frozen single-run teacher baseline**, not a statistically replicated or publication-validated result. Further independent seeds and controlled deployment measurements remain outstanding. The 92.11% held-out test result is a historical reference, not a hyperparameter-tuning target. CNN-004 is now a separately frozen pre-recovery pruned candidate; CNN-005A/B recovery is now complete and separately documented.

This CNN-003B experiment is distinct from **E1**, the ResNet-18 FP32 versus INT8 QDQ quantization validation. E1's existing closure status and findings remain unchanged.

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

## Verified baseline and pruning results

| Metric | CNN-003B teacher | CNN-004 pruned, before recovery |
| --- | ---: | ---: |
| CIFAR-10 validation accuracy (5,000 examples) | 92.38% | **92.18%** |
| Parameters | 23,520,842 | **20,218,770** |
| MACs per 32 × 32 image | 1,311,594,496 | **1,134,338,752** |

CNN-004 reduced parameters by **14.0389%** and MACs by **13.5145%**, with a **0.20 percentage-point** decrease in validation accuracy. These are single-run results, not a multi-seed statistical claim. The teacher's historical final test accuracy was 92.11%; the held-out test set has **not** been used to select CNN-004 or plan recovery.

The fixed split comprises 45,000 CIFAR-10 training examples and 5,000 validation examples, seed 42. The official parent model is CNN-003B, not the earlier CNN-003 teacher.

## CNN-004 freeze and reproducibility

**Decision:** APPROVED — VALIDATED PRE-RECOVERY CANDIDATE. This does not constitute publication-grade validation or final deployment approval.

- Frozen teacher: `checkpoints/CNN-003B/best_model.pt`; SHA-256 `4d3ca6a7b1f0396e4e56024341144d487aa173ace0c7e5d981eaf9879a4ae31a`.
- Frozen pruned weights: [`results/CNN-004/cnn004_pruned_before_recovery_state_dict.pt`](results/CNN-004/cnn004_pruned_before_recovery_state_dict.pt); SHA-256 `595b761259c88e4f656a1a3102ec457d41c57551db301c39111fe4157fb89057`.
- [Formal pre-recovery freeze record](evidence/CNN-004/CNN-004_PRE_RECOVERY_FREEZE_RECORD.md).
- [Pruning implementation](scripts/evaluate_cnn004_pruned.py) and [clean reload verifier](scripts/verify_cnn004_reload.py).
- Freeze record commit: `b36afe6`; frozen checkpoint archive commit: `4435e7c`. The checkpoint is stored in **regular Git**, not Git LFS.

The clean-process verifier was run successfully on the AWS CUDA environment: it verified teacher and student SHA-256 hashes, reconstructed the pruned architecture, loaded the student weights with `strict=True`, and reproduced **92.1800%** validation accuracy, **20,218,770** parameters, and **1,134,338,752** MACs/image. The archived checkpoint contains **36 THOP profiling entries** (`total_ops` / `total_params`); the verifier removes those entries **in memory only** before strict loading. Frozen checkpoint bytes are unchanged.

Run from the repository root with the original environment and dataset available:

```bash
source /opt/pytorch/bin/activate
python -m scripts.verify_cnn004_reload
```

The recorded validation JSON [`results/CNN-004/cnn004_pruned_validation.json`](results/CNN-004/cnn004_pruned_validation.json) was archived in commit [`ef86694`](https://github.com/jyotiaai2020-netizen/architecture-pruning-teacher-healing/commit/ef866947c8480161d6fee1a156746c65a8d56f71). The split manifest and teacher checkpoint must also be available locally for the verifier to run. A GitHub commit and the successful EC2 reload do not independently establish a fresh remote download-and-hash check.

## CNN-005 recovery results and freeze — October 13, 2026

Both CNN-005 arms completed 20 epochs from the **same frozen CNN-004 pruned initialization**, using the same CIFAR-10 45,000/5,000 training/validation split (seed 42), batch size 128, SGD (initial LR 0.01, momentum 0.9, weight decay 0.0005), LR milestones 10/16 (gamma 0.1), and AMP. CNN-005A used supervised cross-entropy; CNN-005B used frozen-teacher knowledge distillation (temperature 4.0, alpha 0.5).

| Model | Parameters | Best validation accuracy | Best epoch | Independent validation |
|---|---:|---:|---:|---|
| CNN-003B teacher | 23,520,842 | 92.38% | 48 | Previously frozen |
| CNN-004 pruned, pre-recovery | 20,218,770 | 92.18% | N/A | Previously frozen |
| **CNN-005A supervised recovery** | **20,218,770** | **92.64%** | 20 | **PASS: 4,632 / 5,000** |
| **CNN-005B teacher-guided recovery** | **20,218,770** | **92.88%** | 20 | **PASS: 4,644 / 5,000** |

The recovered pruned architecture retains the CNN-004 reductions of **14.04% in parameters** and **13.51% in MACs** relative to the teacher. CNN-005B exceeded CNN-005A by **0.24 percentage points** on the frozen validation split. This is a **single-seed observation**, not a statistical significance or causal generalization claim. Neither CNN-005 arm accessed the held-out CIFAR-10 test set.

**Verification and preservation:** Both best checkpoints deserialized, passed strict model reload with 20,218,770 parameters, and reproduced their recorded 5,000-image validation accuracies. SHA-256 hashes were recorded for both best and final checkpoints and both summary JSON files. All four checkpoint copies in `~/research-freeze-backups/CNN-005A/` and `~/research-freeze-backups/CNN-005B/` were hash-verified against the originals on EC2. This is a **same-instance backup**, not an off-instance disaster-recovery copy.

- [CNN-005 recovery comparison and formal research freeze record](evidence/CNN-005/CNN-005_RECOVERY_COMPARISON_AND_FREEZE_RECORD.md) (created on `experiment/cnn005-recovery`, commit `992de696185d311fd2f15b900265d9432ef3359c`; link will resolve on main after branch merge).
- Runner: `scripts/run_cnn005_recovery.py` (on `experiment/cnn005-recovery`).
- EC2 summaries: `results/CNN-005A/summary.json`, `results/CNN-005B/summary.json` (not yet claimed committed to Git).
- Best checkpoint hashes: CNN-005A `b974d77fb6cf6d7b2235ea0ec928a95a4e8085e930357a2b34392bc5a57b7639`; CNN-005B `62eb8ec8a2826848067236fdd85aae10b06a24b3a2af9dbb929aeb16208f065e`.
- Full six-artifact SHA-256 inventory and experimental limitations are in the freeze record. Large CNN-005 checkpoint binaries are **not hosted in GitHub**.

**Freeze scope:** CNN-005A and CNN-005B are research-evidence-frozen for comparison. Off-instance archival, repeated-seed uncertainty, held-out testing under a predeclared final evaluation protocol, and deployment performance remain outstanding.

## Next controlled experiments

**CNN-006 (planned):** establish controlled FP32/FP16 baselines on NVIDIA Tesla T4 and evaluate quantization with a supported INT8 backend. PyTorch 2.12.1+cu130, CUDA/T4, ONNX and ONNX Runtime were observed; TensorRT and Torch-TensorRT were not installed at preflight. ONNX Runtime execution providers still require verification. Compare the teacher, CNN-005A, and CNN-005B under matched accuracy and deployment protocols. Preserve frozen source checkpoints; store quantized outputs separately.

**CNN-007 (planned):** compare combined compression and edge deployment efficiency with measured latency, throughput, memory, complete artifact size, and energy where instrumented. Do not infer real speedup from parameter/MAC reductions alone.

## Environment and evidence governance

- AWS EC2 g4dn.xlarge / NVIDIA Tesla T4; Ubuntu; Python virtual environment `/opt/pytorch`; PyTorch; Torch-Pruning 1.6.0; THOP.
- MLflow is part of the broader research tooling, but complete experiment-tracking coverage should not be inferred from the current freeze artifacts.
- Never overwrite frozen CNN-003B/CNN-004 artifacts, reuse held-out test data for model selection, or stage unrelated E1 governance work.
- Outstanding: off-instance checkpoint archival, independent remote artifact integrity checks, controlled latency/energy benchmarks, multi-seed replication, and CNN-006/007 quantization/deployment experiments.
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

CNN-003B and CNN-004 are frozen for the purposes documented above. CNN-005A/B recovery has completed, passed strict reload and independent validation, and is research-evidence-frozen with verified same-instance checkpoint backups. CNN-006 quantization remains planned. E1 remains an independent FP32-versus-INT8 study in closure validation; its pending paired capability, calibration, robustness, controlled performance/memory, and governance steps are unchanged. No complete EDF score or causal Teacher-Healing benefit has been established.

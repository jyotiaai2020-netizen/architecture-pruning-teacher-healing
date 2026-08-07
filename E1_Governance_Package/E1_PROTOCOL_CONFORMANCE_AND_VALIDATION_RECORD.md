# E1 Protocol Conformance and Validation Record

**Record ID:** `E1-CONFORMANCE-2026-08-06`  
**Experiment:** E1-B — ONNX Runtime Static INT8 PTQ  
**Status:** CONDITIONAL / NOT FROZEN  
**Baseline specification:** `CR-EDF-SPEC-v1.0`

## Chronology disclosure

E1 execution began before the E0 governance package was formally committed.
Accordingly, the E0 package is a retrospective record of the intended baseline
protocol for E1, not proof of prospective authorization. This record performs a
protocol-conformance review of the completed work. Future experiments require
prospective authorization before execution.

## Artifact boundary

| Role | Artifact | Governance treatment |
|---|---|---|
| Training reference | PyTorch CNN-002B FP32 checkpoint | Frozen E0 source checkpoint |
| Runtime control | Exported ONNX FP32 model, including external weights | E1 control artifact |
| Candidate | ONNX Runtime static INT8 Q/DQ model | E1 compression candidate |

Conversion validation, ONNX checker status, graph structure, and prediction
agreement must be reported separately from top-line accuracy.

## Confirmed evidence

| Validation item | Result |
|---|---|
| Calibration population | 512 CIFAR-10 training observations |
| Calibration manifest | 513 lines including header |
| Evaluation population | 10,000 official CIFAR-10 test observations |
| Evaluation manifest | 10,001 lines including header |
| ONNX FP32 accuracy | 93.81% |
| Static INT8 accuracy | 93.82% |
| Observed paired accuracy difference | +0.01 percentage points |
| Capability retention | 100.01% |
| SHA-256 validation | 20/20 listed artifacts passed |
| Corrected evidence commit | `411d6a6` |
| Checksum-validation commit | `30a974e` |

Checksum validation establishes artifact integrity and readability. It does not,
by itself, validate metric correctness, statistical significance, model-size
accounting, robustness, or EDF.

## Corrected protocol deviations

1. An earlier evaluation used only 1,000 test observations.
2. The corrected evaluation uses all 10,000 official test observations.
3. Calibration is separated from evaluation and uses the training partition.
4. Prior efficiency measurements are treated as stale because the INT8 artifact
   was regenerated.
5. FP32 storage must include any external `.onnx.data` weights.

## Statistical validation required

- [ ] Preserve 95% confidence intervals for each model’s clean accuracy.
- [ ] Preserve complete per-class precision, recall, F1, support, accuracy, and confidence intervals.
- [ ] Preserve both confusion matrices.
- [ ] Preserve prediction-level paired FP32/INT8 results for all 10,000 observations.
- [ ] Report McNemar’s test with contingency counts and exact/asymptotic method identified.
- [ ] Report bootstrap 95% CI for the paired accuracy difference.
- [ ] Report bootstrap 95% CI for capability retention.
- [ ] State explicitly whether the observed difference is statistically distinguishable from zero.

## Model and data validation required

- [ ] Reconfirm ONNX checker status for FP32 and regenerated INT8 artifacts.
- [ ] Record Q/DQ node types and counts for the candidate graph.
- [ ] Verify calibration reader consumed exactly 512 unique manifest observations.
- [ ] Verify evaluation manifest contains exactly test indices 0–9999 with no duplicates.
- [ ] Verify all current checksums after any additional evidence is created.
- [ ] Record preprocessing equivalence across PyTorch, ONNX FP32, and INT8 paths.
- [ ] Quantify prediction agreement between PyTorch FP32 and ONNX FP32.

## Robustness conformance required

E1 must be evaluated on the same CIFAR-10-C protocol as E0:

- all 15 standard corruptions;
- severities 1–5;
- all 75 condition pairs;
- 750,000 prediction rows;
- mean corruption accuracy, macro-F1, ECE, and NLL;
- per-corruption and per-severity comparisons; and
- prediction-level failure consistency.

The E1 robustness gate remains open until these artifacts are complete.

## Efficiency and EDF measurements to refresh

All measurements must compare the ONNX FP32 control and regenerated INT8
candidate on the same controlled device and runtime configuration.

| Dimension | Minimum governance requirement |
|---|---|
| Storage | Include ONNX files and all external weight files |
| Latency | Warm-up documented; at least five independent repetitions; median reported |
| Throughput | Same batch sizes, threads, execution provider, and input population |
| Memory | Isolated-process RSS and USS; accelerator memory where applicable |
| Energy | Directly measured, documented proxy, or marked unavailable |
| EDF | Approved weights, targets, normalization bounds, and missing-data rule |
| Capability Efficiency | Recalculate from validated Capability Score and EDF Score |

If energy is unavailable, no silent zero is permitted. Either the composite EDF
remains incomplete or a prospective, approved reweighting amendment must be
recorded before evaluation.

## Conformance determination

| Gate | Current state |
|---|---|
| Evidence integrity | Passed |
| Calibration/test separation | Passed |
| Full clean-test capability | Passed |
| Paired statistical inference | Open |
| Per-class and prediction completeness | Open |
| ONNX/QDQ structural verification | Open |
| CIFAR-10-C robustness | Open |
| Refreshed storage/latency/memory/throughput | Open |
| Energy treatment | Open |
| EDF and Capability Efficiency | Open |

**Determination:** E1 supports full-test capability preservation but is not
eligible for approval or freeze until all mandatory open gates are resolved.

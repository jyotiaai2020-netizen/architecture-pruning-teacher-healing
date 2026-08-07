# E1 Freeze Readiness Checklist

**Candidate:** ONNX Runtime static INT8 Q/DQ  
**Current status:** NOT READY TO FREEZE  
**Decision standard:** `CR-EDF-SPEC-v1.0`

## A. Identity and integrity

- [x] Corrected candidate evidence committed.
- [x] Checksum-validation record committed.
- [x] All 20 listed artifacts passed SHA-256 verification.
- [x] Calibration manifest contains 512 records.
- [x] Evaluation manifest contains 10,000 records.
- [ ] Final candidate checksum is explicitly named in the freeze record.
- [ ] Source revision, environment, ONNX Runtime version, and execution provider are frozen.
- [ ] ONNX checker and Q/DQ graph verification are complete.

## B. Capability

- [x] Full official CIFAR-10 test set evaluated.
- [x] ONNX FP32 accuracy recorded as 93.81%.
- [x] Static INT8 accuracy recorded as 93.82%.
- [x] Capability retention recorded as 100.01%.
- [ ] Accuracy confidence intervals are preserved.
- [ ] Per-class metrics and confidence intervals are preserved.
- [ ] Both confusion matrices are preserved.
- [ ] Paired predictions for all 10,000 observations are preserved.
- [ ] McNemar’s test is reported.
- [ ] Bootstrap CI for paired accuracy difference is reported.
- [ ] Bootstrap CI for capability retention is reported.

## C. Robustness

- [ ] All 15 CIFAR-10-C corruptions are evaluated.
- [ ] All five severity levels are evaluated.
- [ ] All 75 corruption–severity conditions are present.
- [ ] All 750,000 candidate predictions are preserved.
- [ ] Mean corruption-accuracy drop is no more than 2.00 percentage points.
- [ ] No severity level loses more than 4.00 percentage points.
- [ ] Mean ECE increases by no more than 3.00 percentage points.
- [ ] Macro-F1, ECE, NLL, and failure consistency are reported.

## D. Efficiency and EDF

- [ ] FP32 storage includes external `.onnx.data` weights.
- [ ] INT8 storage is recalculated from the regenerated artifact.
- [ ] Compression ratio and storage reduction are recalculated.
- [ ] Latency is rerun with documented warm-up and at least five repetitions.
- [ ] Throughput is rerun under matched settings.
- [ ] Isolated-process RSS and USS are rerun.
- [ ] Accelerator memory is reported when applicable.
- [ ] Energy is measured, supported by an approved proxy, or formally unavailable.
- [ ] EDF is calculated with approved weights and normalization bounds.
- [ ] Capability Efficiency and relative gain are recalculated.
- [ ] At least one EDF dimension improves by 10% or more.
- [ ] No EDF dimension regresses by more than 5%.

## E. Decision control

- [ ] Every mandatory integrity, capability, robustness, and EDF gate passes.
- [ ] Any exception was approved prospectively and documented.
- [ ] Composite score is calculated only after mandatory gates pass.
- [ ] Decision band is applied unambiguously:
  - **Approve:** score ≥ 90.
  - **Approve with conditions:** 85 ≤ score < 90.
  - **Review required:** incomplete measurement or approved exception.
  - **Reject:** integrity/test-contamination failure or non-compensable gate failure.
- [ ] Final E1 freeze record identifies exact commits, artifact checksums, results, and approval date.

## Freeze rule

Do not label E1 “approved,” “scientifically complete,” or “frozen” while any
mandatory item remains open. The permitted current statement is:

> E1 provides full-test capability-preservation support with validated evidence
> integrity. Statistical, robustness, efficiency, and EDF conclusions remain
> conditional pending completion of their respective gates.

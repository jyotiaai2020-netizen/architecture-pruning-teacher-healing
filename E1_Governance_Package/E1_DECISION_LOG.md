# E1 Decision Log

**Experiment:** E1-B — ONNX Runtime Static INT8 PTQ  
**Record date:** 2026-08-06  
**Decision owner:** Principal Researcher  
**Status:** Conditional; validation in progress  
**Baseline:** E0 CNN-002B FP32  
**Candidate evidence path:** `evidence/E1/ONNX_INT8/`

| ID | Decision | Rationale and evidence | Status |
|---|---|---|---|
| DEC-E1-001 | Treat the PyTorch CNN-002B checkpoint, exported ONNX FP32 control, and ONNX Runtime Q/DQ INT8 candidate as distinct artifacts. | Equal accuracy does not establish artifact identity; conversion and prediction agreement require separate validation. | Approved |
| DEC-E1-002 | Use static post-training quantization with Q/DQ representation. | This is the defined E1 deployment candidate and must be verified through ONNX checker status and Q/DQ node counts. | Approved |
| DEC-E1-003 | Use 512 frozen CIFAR-10 training observations for calibration. | Calibration must not consume official test observations. The manifest contains 512 records plus one header. | Validated |
| DEC-E1-004 | Evaluate capability on all 10,000 official CIFAR-10 test observations. | Corrects the earlier 1,000-image limitation and prevents partial-test inference. The manifest contains 10,000 records plus one header. | Validated |
| DEC-E1-005 | Record ONNX FP32 accuracy of 93.81% and static INT8 accuracy of 93.82%. | The observed difference is +0.01 percentage points, equal to one prediction out of 10,000. | Validated |
| DEC-E1-006 | Interpret the result as capability preservation, not improvement. | Statistical paired testing is required to determine whether the one-prediction difference exceeds ordinary prediction variation. | Approved |
| DEC-E1-007 | Preserve the corrected evidence bundle and checksum-validation output in GitHub. | Corrected evidence was integrated through commits `411d6a6` and `30a974e`; all 20 listed artifacts validated successfully. | Validated |
| DEC-E1-008 | Keep source datasets under `data/` untracked while tracking curated experiment evidence under controlled `evidence/E1/` paths. | Separates reproducible evidence from redundant dataset downloads and transient files. | Approved |
| DEC-E1-009 | Do not carry forward efficiency values measured against the superseded INT8 artifact. | Latency, memory, storage, throughput, energy, EDF, and Capability Efficiency must be refreshed against the regenerated candidate. | Approved |
| DEC-E1-010 | Count external `.onnx.data` weights in FP32 storage size. | Excluding external weights understates the FP32 artifact footprint and invalidates compression calculations. | Approved |
| DEC-E1-011 | Require paired inference statistics. | McNemar’s test, bootstrap 95% CI for paired accuracy difference, and bootstrap CI for capability retention are required. | Open |
| DEC-E1-012 | Require full per-class and prediction-level evidence. | Per-class precision, recall, F1, support, accuracy, confidence intervals, confusion matrices, and paired FP32/INT8 predictions must be preserved. | Open |
| DEC-E1-013 | Require E0-aligned CIFAR-10-C robustness evaluation before E1 freeze. | Clean capability preservation alone does not establish preservation under corruption or shift. | Open |
| DEC-E1-014 | Apply `CR-EDF-SPEC-v1.0` prospectively to the corrected artifact, with no silent treatment of missing energy. | EDF conclusions are conditional until controlled measurements are complete. | Open |
| DEC-E1-015 | Classify E1 as CONDITIONAL / NOT FROZEN. | Capability and integrity evidence are sufficient for provisional support, but all non-compensable gates have not been completed. | Approved |

## Current research conclusion

> Static INT8 calibration used 512 frozen CIFAR-10 training observations, and
> capability evaluation used all 10,000 official test observations. ONNX FP32
> achieved 93.81% accuracy and static INT8 achieved 93.82%, yielding 100.01%
> capability retention and an observed difference of +0.01 percentage points.
> This supports full-test capability preservation. It does not establish
> statistically significant improvement, robustness preservation, or EDF benefit
> until the remaining paired, robustness, and controlled-efficiency gates pass.

## Change control

Any later correction must identify the affected decision, reason, authorizer,
changed artifact, old and new checksums, and downstream impact. Existing
evidence must remain recoverable; corrected evidence must use a versioned path
or explicit amendment record.

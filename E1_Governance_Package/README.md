# E1 Governance Package

**Experiment:** E1-B — ONNX Runtime Static INT8 PTQ  
**Governance status:** CONDITIONAL / NOT FROZEN  
**Baseline:** E0 CNN-002B FP32  
**Candidate:** ONNX Runtime static INT8 Q/DQ  
**Evidence path:** `evidence/E1/ONNX_INT8/`  
**Evidence commits:** `411d6a6`, `30a974e`  
**Main integration commits:** `3a1b197`, `4c1a7e9`  
**Governing specification:** `CR-EDF-SPEC-v1.0`

## Purpose

This package governs E1 separately from the immutable E0 baseline. It records the
actual experiment chronology, known protocol corrections, validated evidence,
open scientific gates, and the conditions required before E1 may be frozen or
used as the accepted INT8 deployment candidate.

## Current determination

E1 provides **full-test capability-preservation support**:

- calibration used 512 frozen observations from the CIFAR-10 training partition;
- evaluation used all 10,000 observations in the official CIFAR-10 test partition;
- ONNX FP32 accuracy was 93.81%;
- static INT8 accuracy was 93.82%;
- observed difference was +0.01 percentage points;
- capability retention was 100.01%; and
- all 20 listed evidence artifacts passed SHA-256 verification.

The one-prediction difference is not classified as an improvement unless paired
statistical testing supports that claim. E1 remains unfrozen because robustness,
paired statistical inference, refreshed efficiency measurements, complete model
size accounting, and EDF scoring have not all passed their required gates.

## Package contents

- `E1_DECISION_LOG.md` — decisions, corrections, and current disposition.
- `E1_PROTOCOL_CONFORMANCE_AND_VALIDATION_RECORD.md` — validated facts,
  deviations, evidence integrity, and outstanding gates.
- `E1_FREEZE_READINESS_CHECKLIST.md` — exact requirements for E1 approval and freeze.

E0 governance records are not modified by this package.

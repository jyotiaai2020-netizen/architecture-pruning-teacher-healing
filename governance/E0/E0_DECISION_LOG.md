# E0 Decision Log

**Experiment:** E0 — CNN-002B FP32 Baseline  
**Record date:** 2026-07-29  
**Decision owner:** Principal Researcher  
**Status:** Approved  
**Authoritative evidence commit:** `ad8f55a`  
**Repository integration:** Confirmed in `origin/main` on 2026-07-29  

## Purpose

This log records the decisions that established, evaluated, preserved, and froze
the E0 FP32 reference. It is a governance record and does not replace the
machine-readable evidence.

| ID | Decision | Rationale and evidence | Status |
|---|---|---|---|
| DEC-E0-001 | Use CNN-002B FP32 as the canonical E0 baseline. | Provides a stable teacher/reference checkpoint for all compression experiments. Checkpoint SHA-256: `d106883bd4fda76a9bd6c15ec7d5d398bc8a39d8337a9d7da11dcd817f405e36`. | Approved |
| DEC-E0-002 | Measure clean capability on CIFAR-10. | CIFAR-10 is the training-domain, balanced, 10-class benchmark. Validated clean accuracy: 93.81%. | Approved |
| DEC-E0-003 | Measure robustness on CIFAR-10-C before E1. | Clean accuracy alone cannot establish capability preservation under corruption or distribution shift. | Approved |
| DEC-E0-004 | Evaluate all 15 standard corruptions at severities 1–5. | The complete 75-condition protocol prevents selective reporting. | Approved |
| DEC-E0-005 | Keep CIFAR-10-C evaluation-only. | Prevents training/tuning contamination of the robustness reference. | Approved |
| DEC-E0-006 | Use seed 42, batch size 128, and one CPU thread for the recorded run. | Fixes reproducibility-relevant execution settings. | Approved |
| DEC-E0-007 | Retain prediction-level evidence. | Enables failure consistency, paired comparisons, and later teacher-healing analysis. | Approved |
| DEC-E0-008 | Treat AlexNet-normalized mCE as not calculated. | Reference constants were not configured; raw error, accuracy, F1, ECE, and NLL remain valid. No value may be inferred or fabricated. | Approved |
| DEC-E0-009 | Preserve the complete evidence set with SHA-256 verification. | All five listed evidence files passed checksum validation. | Approved |
| DEC-E0-010 | Preserve evidence off-instance in GitHub. | Commit `ad8f55a` is present in `origin/main`, preventing reliance on ephemeral NVMe storage. | Approved |
| DEC-E0-011 | Leave `data/` and `evidence/` untracked. | They were not part of the reviewed E0 evidence commit and must not enter the freeze by accident. | Approved |
| DEC-E0-012 | Adopt capability–robustness–EDF weighting of 40%–35%–25%. | Capability and robustness are primary preservation goals; EDF rewards efficiency without permitting it to compensate for failed scientific gates. | Approved |
| DEC-E0-013 | Apply non-compensable capability, robustness, integrity, and reproducibility gates before composite scoring. | Prevents an efficient but materially degraded model from being accepted solely because of EDF gains. | Approved |
| DEC-E0-014 | Freeze E0 at evidence commit `ad8f55a`. | The full 75-condition run, 750,000 predictions, checksums, remote integration, and governance review are complete. | Approved |
| DEC-E0-015 | Authorize E1 INT8 PTQ only as a new experiment derived from the frozen E0 baseline. | E1 may add new files and code but must not modify E0 evidence or governance records except through a separately approved amendment. | Approved |

## Recorded E0 results

| Measure | Frozen E0 value |
|---|---:|
| Clean CIFAR-10 accuracy | 93.81% |
| CIFAR-10-C mean corruption accuracy | 71.2695% |
| Mean corruption error | 28.7305% |
| Mean macro-F1 | 71.1279% |
| Mean ECE (15-bin) | 18.6876% |
| Mean negative log-likelihood | 1.381214 |
| Conditions | 75/75 |
| Image evaluations | 750,000 |
| Evaluation runtime | 03:08:30 |

## Change control

Any correction after freeze requires a new amendment record that identifies:
the affected decision, reason, authorizer, changed artifact, old and new
checksums, and effect on downstream experiments. The original frozen evidence
must remain recoverable.

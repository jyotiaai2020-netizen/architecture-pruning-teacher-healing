# E0 Formal Freeze Record

**Freeze ID:** `FREEZE-E0-2026-07-29`
**Experiment:** E0 — CNN-002B FP32 Baseline
**Freeze date:** 2026-07-29
**Decision owner:** Principal Researcher
**Freeze status:** APPROVED
**Evidence commit:** `ad8f55a`
**Evidence integration:** Commit confirmed reachable from `origin/main`
**Scoring specification:** `CR-EDF-SPEC-v1.0`

## Freeze declaration

E0 is formally frozen as the scientific baseline for subsequent quantization,
pruning, and teacher-healing experiments. The freeze is anchored to evidence
commit `ad8f55a`, which contains the completed CIFAR-10-C evidence set and
execution log and has been confirmed in `origin/main`.

The freeze applies to the contents and checksums of the validated E0 evidence.
It does not assert that the untracked `data/` or `evidence/` directories are
part of the scientific record.

## Gate verification

| Freeze gate | Evidence | Result |
|---|---|---|
| Clean capability baseline established | CIFAR-10 clean accuracy 93.81% | Passed |
| Robustness protocol complete | 15 corruptions × 5 severities | Passed |
| Condition completeness | 75 rows and 75 unique pairs | Passed |
| Prediction completeness | 750,000 prediction rows | Passed |
| Aggregate robustness recorded | Accuracy, error, macro-F1, ECE, and NLL | Passed |
| Evidence integrity | Five referenced files reported `OK` under SHA-256 verification | Passed |
| Execution log retained | Added in commit `ad8f55a` | Passed |
| Off-instance preservation | Evidence commit present in GitHub `origin/main` | Passed |
| Scoring method approved | `CR-EDF-SPEC-v1.0` | Passed |
| Decision trail approved | E0 Decision Log | Passed |

## Frozen reference values

| Reference | Frozen value |
|---|---:|
| Checkpoint SHA-256 | `d106883bd4fda76a9bd6c15ec7d5d398bc8a39d8337a9d7da11dcd817f405e36` |
| Clean accuracy | 93.81% |
| Mean corruption accuracy | 71.2695% |
| Mean corruption error | 28.7305% |
| Mean macro-F1 | 71.1279% |
| Mean ECE (15-bin) | 18.6876% |
| Mean NLL | 1.381214 |
| AlexNet-normalized mCE | Not calculated |
| CIFAR-10-C conditions | 75 |
| Image evaluations | 750,000 |
| Seed | 42 |
| Batch size | 128 |
| CPU threads | 1 |
| Runtime | 03:08:30 |

## Protection rule

Files comprising the frozen E0 evidence must not be overwritten, regenerated,
rebased away, or amended in place. A discovered defect requires:

1. a versioned freeze amendment,
2. preservation of the original evidence,
3. identification of changed artifacts and checksums,
4. impact analysis for every dependent experiment, and
5. explicit reauthorization.

## E1 authorization

E1 INT8 post-training quantization is authorized to begin after this governance
package is committed and pushed to a branch reachable from `origin/main`.

E1 must:

- identify E0 commit `ad8f55a` as its baseline;
- use the approved `CR-EDF-SPEC-v1.0` protocol;
- write to new E1 paths and never alter frozen E0 artifacts;
- retain checkpoint, environment, EDF, clean-capability, robustness, and
  prediction-level evidence; and
- remain unfrozen until its own validation and governance gates pass.

## Approval

By approving this record, the Principal Researcher confirms that the listed
gates have been reviewed and that E0 is the immutable reference for subsequent
experiments.

**Approved by:** Principal Researcher
**Approval date:** 2026-07-29
**Decision:** Freeze E0 and authorize controlled E1 INT8 PTQ execution

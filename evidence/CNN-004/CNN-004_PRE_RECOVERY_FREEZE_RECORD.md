# CNN-004 — Pre-Recovery Candidate Freeze Record

## Freeze decision

**APPROVED — VALIDATED PRE-RECOVERY CANDIDATE**

Recorded (UTC): 2026-10-12T23:04:13.697759+00:00

This approval applies to the single CNN-004 pruned candidate
evaluated before any recovery training.

It does not constitute final model approval or publication-level
statistical validation.

## Parent baseline

- Experiment: CNN-003B
- Architecture: CIFAR-adapted ResNet-50
- Teacher checkpoint: `checkpoints/CNN-003B/best_model.pt`
- Teacher SHA-256: `4d3ca6a7b1f0396e4e56024341144d487aa173ace0c7e5d981eaf9879a4ae31a`
- Teacher best validation accuracy: 92.38%
- Teacher final test accuracy: 92.11% (historical baseline only)

## Pruning configuration

- Method: structured channel pruning
- Library: Torch-Pruning 1.6.0
- Importance: L2 magnitude
- Target eligible-channel pruning ratio: 0.20
- Iterative steps: 1
- Global pruning: false
- Root modules: Conv2d
- Protected: classifier, stem convolution, 1x1 convolutions
- Recovery training: none

## Verified results

| Metric | Before | After |
|---|---:|---:|
| Parameters | 23,520,842 | 20,218,770 |
| MACs | 1,311,594,496 | 1,134,338,752 |
| Validation accuracy | 92.38% | 92.18% |

- Parameter reduction: 14.0389%
- MAC reduction: 13.5145%
- Validation accuracy change: -0.20 percentage points
- Validation images: 5,000
- Validation split seed: 42

## Evidence and artifact integrity

- Split manifest: `evidence/CNN-003B/cifar10_train45k_val5k_seed42.json`
- Split manifest SHA-256: `1c10c8261334ccc8cc3035b6fb3cfc7083b6aee45d3913971eaff11044e1d4bf`
- Pruned weights: `results/CNN-004/cnn004_pruned_before_recovery_state_dict.pt`
- Pruned weights SHA-256: `595b761259c88e4f656a1a3102ec457d41c57551db301c39111fe4157fb89057`
- Validation result: `results/CNN-004/cnn004_pruned_validation.json`
- Validation result SHA-256: `005d435a5d7ed2d060ced034a0b37a712ef1938bb8380fbe3443819600fb5c2d`
- Smoke-test result: `checkpoints/CNN-004/cnn004_pruning_smoke.json`
- Smoke-test result SHA-256: `46fcf1298aa537730d87f307abfe6e822a3546c5b4c35573440d6d62c25c4b1b`
- Original smoke model SHA-256: 99918cf0f463da6598738f3dadffcdc9c82415840d9c576b567df178681e6730

## Reproducibility limitations

The validation candidate was reconstructed by reapplying
the recorded pruning procedure to the frozen teacher.

Its parameter and MAC counts match the smoke-test model,
but tensor-level equivalence between the original smoke
artifact and the evaluated candidate has not been established.

The saved state dictionary requires reconstruction of the
pruned architecture before loading.

A clean-process reload test, off-instance backup, controlled
latency benchmark, and multi-seed replication remain pending.

## Governance controls

1. Do not modify the frozen CNN-003B teacher.
2. Do not overwrite CNN-004 pre-recovery weights or results.
3. Use the same training split and validation protocol for recovery.
4. Do not use the held-out test set for recovery model selection.
5. Preserve E1 governance evidence and unrelated working changes.
6. Record fine-tuning and teacher-guided recovery separately.

## Next experiments

- CNN-005A: standard supervised fine-tuning
- CNN-005B: teacher-guided recovery
- Compare under matched training and evaluation budgets.

## Freeze classification

Validated pre-recovery candidate.

Formal repository commit and remote backup remain separate
verification steps.

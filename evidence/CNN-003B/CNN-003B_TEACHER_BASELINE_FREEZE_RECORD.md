# CNN-003B Teacher Baseline Freeze Record

## 1. Experiment identification

- Experiment ID: CNN-003B
- Model: CIFAR-adapted ResNet-50
- Dataset: CIFAR-10
- Status: Baseline artifact frozen; single-run evidence
- Freeze scope: Teacher reference for CNN-004 structured pruning
- Source commit: c08e6ce5908d3b7abaceb512936a1bdb3fdd0835
- MLflow run: edc07a6c32b0485db5db7e53cff30082

## 2. Training and evaluation protocol

- Training observations: 45,000
- Validation observations: 5,000
- Official test observations: 10,000
- Split seed: 42
- Epochs: 50
- Selection metric: Validation accuracy
- Best epoch: 48
- Final test evaluation: After checkpoint selection
- Device: NVIDIA Tesla T4 GPU

Dataset partition integrity was verified:
- No duplicate indices
- No train/validation overlap
- Complete coverage of the official 50,000 training observations

## 3. Observed baseline results

| Metric | Value |
|---|---:|
| Best validation accuracy | 92.38% |
| Final test accuracy | 92.11% |
| Parameters | 23,520,842 |
| Reported model size | 90.03 MiB |
| Training wall-clock time | 2,662.68 seconds |

These are single-run observations, not statistical confidence bounds.

## 4. Frozen teacher artifact

Approved reference:

checkpoints/CNN-003B/best_model.pt

SHA-256:

4d3ca6a7b1f0396e4e56024341144d487aa173ace0c7e5d981eaf9879a4ae31a

The final-epoch checkpoint is preserved separately and is
not the designated CNN-004 teacher reference.

## 5. Reproducibility evidence

- scripts/run_cnn_experiment_cnn003b_validation.py
- configs/cnn/cnn003b_validation_50ep.yaml
- metadata/splits/cifar10_train45k_val5k_seed42.json
- results/CNN-003B/summary.json
- evidence/CNN-003B/checkpoint_sha256.txt
- evidence/CNN-003B/cnn003b_teacher_50ep.log

## 6. Backup verification

A matching copy of the selected checkpoint was verified at:

/home/ubuntu/research-freeze-backups/CNN-003B/best_model.pt

Both checkpoint files produced the same SHA-256 fingerprint.

This backup is on the same EC2 instance and does not
constitute independent off-instance disaster recovery.

## 7. Freeze controls

1. Do not modify or overwrite the frozen teacher checkpoint.
2. CNN-004 must use a separate experiment identifier and output directory.
3. Record all pruning configurations and random seeds.
4. Compare models under consistent evaluation conditions.
5. Use validation data for development and model selection.
6. Do not repeatedly tune against held-out test results.
7. Preserve historical CNN-003 and E1 evidence.
8. Record deviations through a new versioned amendment.

## 8. Research limitations

- One completed training seed.
- No multi-seed confidence intervals established.
- Deployment latency, energy, and robustness comparisons
  are not established by this baseline freeze.
- Freeze approval is limited to the identified teacher artifact,
  not the entire research program.

## 9. Freeze decision

Decision: APPROVED — BASELINE ARTIFACT FREEZE

CNN-003B best_model.pt is designated as the immutable
teacher reference for CNN-004 structured pruning.

This decision does not certify statistical reproducibility
or completion of the wider research program.

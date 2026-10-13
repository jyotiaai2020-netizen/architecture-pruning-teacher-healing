# Architecture Pruning and Teacher Healing

A reproducible research framework for evaluating whether architecture-aware simplification, structured pruning, knowledge distillation, quantization, and teacher-guided recovery can produce compact neural networks with acceptable capability and a lower Effective Deployment Footprint (EDF).

## Research objective

Evaluate model capability alongside parameter count, multiply-accumulate operations (MACs), checkpoint size, inference latency, memory, energy (where measured), and deployment constraints. The intended decision criterion is **Capability Efficiency = Capability Score / EDF Score**, subject to minimum capability requirements. A complete EDF evaluation has **not** yet been performed.

## Experiment status (12 October 2026)

| Experiment | Purpose | Status |
| --- | --- | --- |
| CNN-001 | TinyCNN framework validation | Historical framework experiment |
| CNN-002 | ResNet-18 student baseline | Historical baseline |
| CNN-003 | Initial ResNet-50 teacher | Historical; not the validation-corrected teacher used for CNN-004 |
| CNN-003B | CIFAR-10 ResNet-50 teacher with train/validation separation | **Frozen baseline** |
| CNN-004 | Structured channel pruning, before recovery | **Frozen, checkpoint archived, clean reload passed** |
| CNN-005A | Supervised fine-tuning of CNN-004 | Planned; not trained |
| CNN-005B | Teacher-guided knowledge-distillation recovery of CNN-004 | Planned; not trained |
| CNN-006 | Quantization and compression ablations | Planned; protocol to be finalized |
| CNN-007 | Combined compression and edge/deployment evaluation | Planned; protocol to be finalized |

CNN-005A/B and CNN-006/007 are working experiment labels for the next controlled studies, not completed results. Earlier CNN-005/006/007 labels described conceptual phases; this table clarifies the current operational plan.

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

The recorded validation JSON (`results/CNN-004/cnn004_pruned_validation.json`) is referenced in the freeze record but **is not yet committed on this branch**. The split manifest and teacher checkpoint must also be available locally for the verifier to run. A GitHub commit and the successful EC2 reload do not independently establish a fresh remote download-and-hash check.

## Next controlled experiments

**CNN-005A:** supervised fine-tuning from the frozen CNN-004 state dict.

**CNN-005B:** teacher-guided recovery from the **same frozen CNN-004 state dict**, with CNN-003B frozen in evaluation mode. Match train/validation split, seed, optimizer, learning-rate schedule, augmentation, training budget, and checkpoint-selection rules across both arms. The only intended treatment difference is the teacher-distillation objective. Select using validation data; reserve the held-out test set for final evaluation.

**CNN-006/007:** evaluate quantization-only and pruning-plus-recovery-plus-quantization controls, followed by measured deployment comparisons. Do not assume reduced MACs or lower numerical precision automatically improve GPU latency or energy.

## Environment and evidence governance

- AWS EC2 g4dn.xlarge / NVIDIA Tesla T4; Ubuntu; Python virtual environment `/opt/pytorch`; PyTorch; Torch-Pruning 1.6.0; THOP.
- MLflow is part of the broader research tooling, but complete experiment-tracking coverage should not be inferred from the current freeze artifacts.
- Never overwrite frozen CNN-003B/CNN-004 artifacts, reuse held-out test data for model selection, or stage unrelated E1 governance work.
- Outstanding: independent remote artifact integrity check, controlled latency/energy benchmarks, multi-seed replication, and recovery/quantization experiments.

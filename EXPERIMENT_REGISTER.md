# Master experiment register

Updated 2026-10-28. This is a navigational register, not a substitute for individual frozen evidence.

| ID | Role / lineage | State supported by available records | Canonical records / next gate |
|---|---|---|---|
| CNN-001 | TinyCNN pipeline validation | Historical completed experiment | Confirm archival completeness |
| CNN-002 / CNN-002B | ResNet-18 student baselines | Historical completed experiments | Preserve checkpoint and configuration lineage |
| CNN-003 | Earlier teacher protocol | Historical; superseded for pruning comparisons | Do not use as CNN-004 official parent |
| CNN-003B | CIFAR-adapted ResNet-50 teacher | Frozen single-run baseline | `evidence/CNN-003B/`; replication pending |
| CNN-004 | Structured channel pruning before recovery | Approved validated pre-recovery candidate, not publication-level validation | [Freeze record](evidence/CNN-004/CNN-004_PRE_RECOVERY_FREEZE_RECORD.md); historical freeze text includes pending steps, later README reports clean reload passed |
| CNN-005A | Supervised recovery control | Earlier root README said planned; current branch status requires independent verification | Do not assert completion solely from CNN-006B lineage |
| CNN-005B | Teacher-guided recovery source for CNN-006B | Source model referenced by verified CNN-006B files; complete training status not independently audited here | Inspect training logs, checkpoints and evaluation provenance before claiming causal efficacy |
| CNN-006B | CNN-005B-derived FP32 vs symmetric INT8 Q/DQ v3 TensorRT comparison | Measurements preserved; 5,000-sample validation, latency and energy evidence; **not formally frozen** | [Results](results/CNN-006B/README.md), [protocol](evidence/CNN-006B/EXPERIMENT_PROTOCOL.md), [RQ7 article](paper/RQ7_INT8_T4_ARTICLE_DRAFT.md) |
| CNN-007 | Combined methods and deployment evaluation | Planned in historical overview | Define controlled design before execution |
| E1 | Separate ResNet-18 ONNX Runtime FP32 vs static INT8 Q/DQ | Conditional, not frozen | [E1 governance](E1_Governance_Package/README.md) |

## CNN-006B controlled comparison

| Measurement | FP32 | INT8 v3 | Boundary |
|---|---:|---:|---|
| Accuracy on 5,000 validation observations | 92.88% | 92.62% | Not final test |
| TensorRT engine size (bytes) | 81,685,028 | 21,865,844 | Engine, not full installation |
| GPU-stream latency (ms, median of round medians) | 1.923296 | 0.663472 | CUDA events, no host transfers |
| Observed GPU allocation (MiB) | 104 | 36 | Snapshot, not peak |
| Gross GPU energy (mJ/inference, median across rounds) | 134.444 | 34.0846 | NVML GPU only; includes idle |

## Publication and governance gates

- Verify reproducibility from clean environment and independently downloaded artifacts.
- Report paired accuracy uncertainty and confidence intervals; avoid calling non-significance equivalence.
- Evaluate class-level metrics, calibration, robustness and independent repetitions.
- Report full software/hardware provenance and raw-to-table calculations.
- Separate quantitative result, scientific interpretation, and explicit freeze decision.
- Keep RQ6 teacher-healing claims separate from RQ7 post-training quantization effects.

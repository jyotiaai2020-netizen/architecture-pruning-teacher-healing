# Master research questions and evidence status

Updated 2026-10-28. This register distinguishes the program-level questions from experiment-specific evidence. The E1 results and CNN-006B results use **different models, backends, and evaluation sets** and must not be pooled.

| ID | Research question | Relevant evidence | Current conclusion / missing gate |
|---|---|---|---|
| RQ1 | Can INT8 quantization materially reduce complete deployable model size? | E1 FP32/INT8 ONNX; CNN-006B ONNX and TensorRT engines | Supported for measured artifacts: E1 74.83% size reduction; CNN-006B TensorRT engine 73.23% smaller. Generalization open. |
| RQ2 | Can quantization preserve clean evaluation accuracy? | E1: 93.81% vs 93.82% on 10,000 official test examples; CNN-006B: 92.88% vs 92.62% on 5,000 validation examples | Observed differences documented; equivalence and paired uncertainty not established. CNN-006B uses validation, not an independently reserved final test. |
| RQ3 | Are class-level predictive quality and calibration preserved? | Aggregate and prediction agreement for CNN-006B; E1 partial capability evidence | Open: precision, recall, macro-F1, NLL, ECE, Brier, confusion matrices, uncertainty. |
| RQ4 | Is robustness preserved under corruption or distribution shift? | Planned CIFAR-10-C protocol | Open: matched corruption/severity and shift evaluations. |
| RQ5 | Does compression improve deployment efficiency in a controlled environment? | E1 controlled CPU benchmarking pending; CNN-006B measured T4 GPU-stream latency/memory/energy | **Partially supported for CNN-006B's T4 measurement scope**; end-to-end, CPU, portability and replication open. |
| RQ6 | Does teacher-guided healing causally improve capability beyond ordinary recovery/quantization? | CNN-005A supervised vs CNN-005B teacher-guided recovery lineage; CNN-006B uses a CNN-005B-derived model | **Not established** by CNN-006B. Requires matched controlled intervention/ablation, appropriate baselines, multiple seeds and paired analyses. |
| RQ7 | Does INT8 quantization provide measured hardware-efficiency improvements beyond model-size reduction? | [CNN-006B results](results/CNN-006B/README.md), [protocol](evidence/CNN-006B/EXPERIMENT_PROTOCOL.md), [article](paper/RQ7_INT8_T4_ARTICLE_DRAFT.md) | **Observed T4-specific support**: 2.90× GPU-stream speedup, 65.38% lower observed GPU allocation, 74.65% lower gross GPU energy/inference, with −0.26 pp validation accuracy. Publication-level inference and replication pending. |

## Claim hierarchy

1. **Recorded observation**: measurement directly represented in a source file.
2. **Within-setup comparative finding**: measured FP32 vs INT8 contrast under the specified hardware, backend, dataset and workload.
3. **Replicated inference**: independent repetitions, statistical uncertainty and robustness gates; not yet established for CNN-006B.
4. **General deployment recommendation**: requires additional devices, end-to-end workloads, operating conditions, security and cost.
5. **Teacher-healing causal contribution**: requires a separate controlled intervention; not supported by post-training INT8 comparison alone.

The proposed Effective Deployment Footprint (EDF) and Capability Efficiency remain **uncomputed** until a declared normalization, required metrics and missing-data policy are satisfied.

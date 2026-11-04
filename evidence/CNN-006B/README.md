# CNN-006B evidence guide

**State:** documentation and validation in progress; formal experiment freeze **not approved**.

## What is evidence, and what is a result?

- [Results](../../results/CNN-006B/README.md) report **what was measured**.
- This folder explains **how measurements were obtained, traced, verified and governed**.
- The raw JSON files remain in `results/CNN-006B/validation/`; this folder links to them instead of copying them.

## Verified records

- [Retrospective experiment protocol](EXPERIMENT_PROTOCOL.md).
- [FP32 validation JSON](../../results/CNN-006B/validation/cnn005b_tensorrt_fp32_validation.json).
- [INT8 validation JSON](../../results/CNN-006B/validation/cnn005b_tensorrt_int8_v3_validation.json).
- [Latency JSON](../../results/CNN-006B/validation/tensorrt_fp32_int8_v3_latency.json).
- [Energy JSON](../../results/CNN-006B/validation/tensorrt_fp32_int8_v3_energy.json).
- [Calibration selection JSON](../../results/CNN-006B/validation/int8_calibration_indices.json).
- [Original artifact checksums](../../results/CNN-006B/ARTIFACT_MANIFEST.sha256).

## Measurement integrity

The existing results narrative reports eight draft-release binary artifacts independently downloaded and compared by SHA-256 (8/8 pass). This does not establish an independently replicated experiment. The manifest includes 13 entries, including validation JSON files. Release assets use flat names; the manifest uses original relative paths.

## Remaining governance records

- Full environment manifest with package versions, driver, CUDA, hardware, and build options.
- Detailed claim-to-file validation and performance audit.
- Independent clean-environment rerun and cross-session measurement.
- Paired uncertainty, class-level metrics, calibration and robustness.
- Formal freeze decision, explicitly approved or deferred.

See [RQ7 working article](../../paper/RQ7_INT8_T4_ARTICLE_DRAFT.md) and [research decision log](../../RESEARCH_DECISION_LOG.md). No final freeze decision has been granted.

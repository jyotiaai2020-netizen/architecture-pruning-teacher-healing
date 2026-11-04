# Research navigation and document ownership

Updated: 2026-10-28. Scope: research branch `experiment/cnn006-quantization`.

## Start here

1. [Main research overview](README.md) — objectives, research questions, experiment history, and overall limitations.
2. [Research questions and claim status](RESEARCH_QUESTIONS.md) — RQ1–RQ7, supporting experiments, evidence gaps, and claim boundaries.
3. [Master experimental register](EXPERIMENT_REGISTER.md) — experimental lineage, outputs, status, and next gates.
4. [Research decision log](RESEARCH_DECISION_LOG.md) — append-only, dated cross-experiment decisions; E1 has its own historical [decision log](E1_Governance_Package/E1_DECISION_LOG.md).
5. [RQ7 standalone article draft](paper/RQ7_INT8_T4_ARTICLE_DRAFT.md) — publication-oriented manuscript, **not yet submitted or independently validated**.

## Folder ownership

| Location | Owns | Does not own |
|---|---|---|
| `scripts/` | Executable experimental implementations | Interpretation of results |
| `configs/` | Run configurations | Measurement outputs |
| `results/<experiment>/` | Raw or derived measurements, validation JSON, artifact manifest, concise findings | Formal freeze approval |
| `evidence/<experiment>/` | Protocols, traceability, verification, environment, reproducibility and governance decisions | Redundant copies of large measurement datasets |
| `paper/` | Manuscript drafts and publication figures | Original measurement records |
| `E1_Governance_Package/` | E1-specific decision and conformance history | CNN-006B governance |

## CNN-006B quick links

- [Experimental protocol](evidence/CNN-006B/EXPERIMENT_PROTOCOL.md) — retrospective methodology record.
- [Evidence guide](evidence/CNN-006B/README.md) — sources, integrity, and open gates.
- [Results summary](results/CNN-006B/README.md) — quantitative comparison and measurement boundaries.
- [Checksum manifest](results/CNN-006B/ARTIFACT_MANIFEST.sha256) — 13 original artifact hashes.
- [RQ7 article draft](paper/RQ7_INT8_T4_ARTICLE_DRAFT.md) — research narrative and future validation.
- Binary artifacts were reported as downloaded and hash-checked from a **draft** GitHub release. A draft is not a publicly published archival release.

## Navigation and integrity rules

- Preserve existing paths and frozen artifacts; add cross-links rather than moving experiment data without a migration record.
- Treat model variants and experiment IDs as distinct: E1 is a separate ResNet-18/ONNX Runtime study, not CNN-006B.
- Never equate retrospective protocol documentation with preregistration.
- Never infer formal freeze, teacher-healing efficacy, cross-device generalization, or a complete EDF score from CNN-006B.
- Correct historical claims through explicit amendments and a decision entry rather than silently rewriting raw evidence.

# Cross-experiment research decision log

Established 2026-10-28. Scope: repository-level navigation and CNN-006B decisions. This is an **append-only research governance log**. It does not supersede [E1's own historical decisions](E1_Governance_Package/E1_DECISION_LOG.md) or [CNN-004's freeze record](evidence/CNN-004/CNN-004_PRE_RECOVERY_FREEZE_RECORD.md).

| Decision ID | Date | Decision / rationale | Evidence | Status |
|---|---|---|---|---|
| RD-001 | 2026-10-28 | Keep `results/` for measurement outputs and `evidence/` for protocol, verification, lineage, and governance. Avoid duplicate large datasets. | [Research index](RESEARCH_INDEX.md) | Documentation policy |
| RD-002 | 2026-10-28 | Register RQ7 as a separate T4 hardware-efficiency research question; distinguish it from RQ5's wider deployment-efficiency objective. | [RQ register](RESEARCH_QUESTIONS.md), [CNN-006B results](results/CNN-006B/README.md) | Documentation policy |
| RD-003 | 2026-10-28 | Interpret CNN-006B as **post-training INT8 quantization**, not a teacher-healing treatment test; CNN-005B source lineage does not isolate the healing effect. | [Protocol](evidence/CNN-006B/EXPERIMENT_PROTOCOL.md) | Interpretation restriction |
| RD-004 | 2026-10-28 | Accept measured T4 GPU-stream latency, observed allocation, and gross GPU energy only within their recorded boundaries. | [Results](results/CNN-006B/README.md) | Measured, scope-limited |
| RD-005 | 2026-10-28 | Keep CNN-006B as **evidence preserved / formal freeze pending**; no preregistration, independent replication, or publication approval is implied. | [Protocol](evidence/CNN-006B/EXPERIMENT_PROTOCOL.md) | Pending gates |
| RD-006 | 2026-10-28 | Treat the eight binary assets as hash-verified in the existing **draft** GitHub release; do not represent that draft as published or permanent archival storage. | [Checksum manifest](results/CNN-006B/ARTIFACT_MANIFEST.sha256), [results narrative](results/CNN-006B/README.md) | Draft preservation |
| RD-007 | 2026-10-28 | Do not calculate final EDF / Capability Efficiency from incomplete governed dimensions. | [Main README](README.md) | Open |
| RD-008 | 2026-10-28 | Draft a standalone RQ7 paper, clearly marked **working manuscript**, without invented literature, p-values, confidence intervals, or unverified claims. | [Article draft](paper/RQ7_INT8_T4_ARTICLE_DRAFT.md) | Draft only |

## Change procedure

For future decisions, append a new row with UTC date, owner/reviewer when known, affected experiment, source commit or artifact hash, rationale, disposition, and downstream impacts. Do not erase a historical decision; supersede it explicitly.

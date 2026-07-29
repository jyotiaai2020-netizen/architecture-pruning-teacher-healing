# Approved Capability–Robustness–EDF Scoring Specification

**Specification ID:** `CR-EDF-SPEC-v1.0`
**Approval date:** 2026-07-29
**Baseline:** E0 CNN-002B FP32
**Baseline evidence commit:** `ad8f55a`
**Status:** Approved for E1 and subsequent controlled comparisons

## 1. Objective

This specification determines whether a compressed or teacher-healed model
preserves useful capability and robustness while improving the Environmental
Deployment Footprint (EDF). E0 is the fixed reference. Every candidate must be
evaluated using the same datasets, preprocessing, condition set, measurement
definitions, and controlled benchmarking protocol.

## 2. Weighted score

For a candidate variant \(v\):

\[
S_v = 0.40C_v + 0.35R_v + 0.25E_v
\]

where:

- \(C_v\) = capability score, normalized to E0
- \(R_v\) = robustness score, normalized to E0
- \(E_v\) = EDF improvement score, normalized to the approved EDF targets

The weighted score is calculated only after every mandatory gate passes.

## 3. Capability score (40%)

Capability uses clean CIFAR-10 accuracy:

\[
C_v = 100\times\min\left(1,\frac{A^{clean}_v}{A^{clean}_{E0}}\right)
\]

Frozen E0 clean accuracy is 93.81%. Values above the baseline are capped at 100
for the preservation score and reported separately as gains.

**Mandatory capability gate:** the candidate clean-accuracy drop must be no more
than 1.00 percentage point from E0, unless an explicitly approved experiment
protocol defines a stricter bound.

## 4. Robustness score (35%)

Robustness uses unweighted mean accuracy across the same 15 CIFAR-10-C
corruptions and five severity levels:

\[
R_v = 100\times\min\left(1,\frac{A^{corr}_v}{A^{corr}_{E0}}\right)
\]

Frozen E0 mean corruption accuracy is 71.2695%. Macro-F1, ECE, NLL,
per-severity results, per-corruption results, and prediction-level failure
consistency remain required diagnostic measures; they are not silently folded
into the primary accuracy ratio.

**Mandatory robustness gates:**

1. Mean corruption-accuracy drop must be no more than 2.00 percentage points.
2. No severity level may lose more than 4.00 percentage points of mean accuracy.
3. Mean ECE may not increase by more than 3.00 percentage points.
4. All 75 corruption–severity conditions and 750,000 predictions must be present.

## 5. EDF score (25%)

EDF measures deployability improvements using five equally weighted dimensions:

| EDF dimension | Weight inside EDF |
|---|---:|
| Model/storage footprint reduction | 20% |
| Peak memory reduction | 20% |
| Median latency reduction | 20% |
| Energy per inference reduction | 20% |
| Throughput increase | 20% |

For a reduction metric \(x\):

\[
I_x = 100\times\operatorname{clip}
\left(\frac{x_{E0}-x_v}{x_{E0}\times T_x},0,1\right)
\]

For throughput:

\[
I_{throughput} = 100\times\operatorname{clip}
\left(\frac{x_v-x_{E0}}{x_{E0}\times T_x},0,1\right)
\]

The approved target improvement \(T_x\) is 25% for each EDF dimension. Thus,
meeting a 25% improvement receives 100 for that dimension; smaller improvements
receive proportional credit; regression receives zero. EDF is:

\[
E_v = 0.20\sum I_x
\]

**Mandatory EDF gate:** at least one measured EDF dimension must improve by 10%
or more, no dimension may regress by more than 5%, and all measurements must use
the controlled E0 benchmark method and device.

## 6. Integrity and reproducibility gates

Before a score is valid:

1. The exact candidate checkpoint and source revision must be identified.
2. Required artifacts must pass SHA-256 verification.
3. Dataset, seed, preprocessing, and evaluation protocol must match E0.
4. No CIFAR-10-C example may be used for training, tuning, or model selection.
5. Raw condition and prediction evidence must be retained.
6. Failed or excluded runs must be disclosed; selective condition reporting is prohibited.

Failure of any mandatory gate results in `REJECT` or `REVIEW REQUIRED`,
regardless of the composite score.

## 7. Decision bands

| Outcome | Rule |
|---|---|
| Approve | All gates pass and composite score is at least 90 |
| Approve with conditions | All gates pass and score is 85–89.99, with documented remediation |
| Review required | Score is below 85, or a preapproved exception requires scientific review |
| Reject | Any integrity gate fails, test contamination occurs, or a non-compensable performance gate fails |

## 8. E0 normalization

E0 defines the reference values and is assigned:

| Dimension | E0 reference score |
|---|---:|
| Capability | 100 |
| Robustness | 100 |
| EDF | 0 improvement credit |

E0 is not ranked as a compression candidate. It is the denominator and
scientific control against which E1 and later variants are assessed.

## 9. Amendment control

Weights, gates, targets, or formulas must not be changed after viewing a
candidate's results. Any revision requires a versioned, prospective amendment
approved before the affected experiment is evaluated.

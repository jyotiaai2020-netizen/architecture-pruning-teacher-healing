#!/usr/bin/env bash
# Run only on the original NVIDIA Tesla T4 machine after the release assets
# have been restored to results/CNN-006B/ and Python deps installed.
# This executes 3 separate Python-process sessions. It does NOT claim
# separate host reboots or laboratory-independent replication.
set -Eeuo pipefail

cd "$(git rev-parse --show-toplevel)"
if ! command -v nvidia-smi >/dev/null; then
  echo "BLOCKED: nvidia-smi not found; cannot run a Tesla T4 experiment." >&2; exit 2
fi
if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'T4'; then
  echo "BLOCKED: no NVIDIA T4 GPU visible." >&2; exit 2
fi
for f in results/CNN-006B/cnn-005b_fp32.plan \
         results/CNN-006B/cnn-005b_int8_trt_v3.plan \
         results/CNN-006B/validation/cnn005b_fp32_reference.npz; do
  test -s "$f" || { echo "BLOCKED: missing asset $f" >&2; exit 2; }
done
echo "Preflight passed: model binaries found, NVIDIA T4 visible."
echo "Git commit: $(git rev-parse HEAD)"
echo "Experimental note: fresh OS/GPU reboots and workload exclusivity still"
echo "require independent verification by the operator."

# Capture once before the run; the capture script will refuse overwrites.
if [[ -e publication/rq7_2026_11/environment_manifest_t4.json ]]; then
  echo "Existing environment manifest retained; refusing to overwrite it."
else
  python publication/rq7_2026_11/capture_environment.py
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
for i in 1 2 3; do
  echo "=== Process-session $i/3 ==="
  date -u +'%Y-%m-%dT%H:%M:%SZ'
  nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,power.draw \
    --format=csv
  # Dedicated output directory; the Python runner refuses overwrites.
  python publication/rq7_2026_11/benchmark_representative_energy.py \
    --session-id "${stamp}_session${i}" --rounds 3 --warmups 100
  # Cooling/wait period reduces immediate carry-over; it cannot
  # guarantee full thermal/clock independence.
  if [[ "$i" -lt 3 ]]; then sleep 90; fi
done

python publication/rq7_2026_11/analyze_independent_sessions.py \
  --parent results/CNN-006B/confirmatory --minimum-sessions 3
python publication/rq7_2026_11/analyze_archived_evidence.py
echo "Execution completed. Review results; do NOT mark independent-device"
echo "replication proven without separate host/process/thermal provenance."
echo "Do not git add . or publish derived input NPZ. Stage only reviewed JSON."

#!/usr/bin/env bash
# Re-run text2sql 0.6b plain through the production API at BATCH=32, queued
# behind the in-flight variant chain (the L4 serializes inference and the
# per-IP rate limit would tangle parallel runs).
#
# Purpose: batch-64 (production default) scored 0.76606 vs the local batch-32
# V56A iter-3 number 0.77069. The model is batch-invariant post-e31bf066
# (cos > 0.9999, batch 8 vs 32 agree to 0.0005), so the expectation is a
# noise-floor delta; this run measures it directly on the served stack.
set -uo pipefail

cd "$(dirname "$0")/.."
LOG=logs/text2sql_0p6b_b32_$(date +%Y%m%d).log
mkdir -p logs
exec >>"$LOG" 2>&1

echo "[b32-queue] started $(date '+%F %T'); waiting for the variant chain to finish"
waited=0
while pgrep -f 'run_text2sql_variants|scripts/run_coir.py' >/dev/null 2>&1; do
  waited=$((waited + 1))
  if [ "$waited" -ge 2880 ]; then
    echo "[b32-queue] gave up after 48h"
    exit 1
  fi
  sleep 60
done
echo "[b32-queue] chain finished; starting 0.6b plain BATCH=32 $(date '+%F %T')"

set -a; source /home/shawkatkabbara/memory/.env; set +a
export PAPR_API_KEY="$TEST_X_USER_API_KEY"
ONLY=SyntheticText2SQL BATCH=32 \
  /home/shawkatkabbara/memory/.venv/bin/python scripts/run_coir.py
echo "[b32-queue] DONE rc=$? $(date '+%F %T')"

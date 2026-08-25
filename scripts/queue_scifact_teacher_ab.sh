#!/usr/bin/env bash
# SciFact both-sides teacher A/B through the production API, queued behind the
# SOQA/CFMT chain (the L4 serializes inference; parallel runs would tangle the
# per-IP rate limit).
#
# WHY THIS RUN: the text2sql reasoning runs were queries-only -- the teacher's
# predictive extraction met the student's literal doc bands, a mismatched
# comparison. This is the consistent configuration on the cheapest possible
# corpus (5,183 docs + 300 queries = ~5.5k teacher calls per tier):
#   1. 0.6b plain          (control)
#   2. 0.6b luna  (swift/medium), teacher on BOTH sides
#   3. 0.6b terra (balanced/medium), teacher on BOTH sides
# PAPR_DUMP_DIR persists every returned vector, so each run also yields the
# full band-mass (p_m) surface offline by block re-weighting -- if the teacher
# helps but the learned gate under-weights it, the surface is the evidence for
# a per-task/per-tier mass-table entry (the iter-3 gate-offset mechanism).
set -uo pipefail

cd "$(dirname "$0")/.."
LOG=logs/scifact_teacher_ab_$(date +%Y%m%d).log
mkdir -p logs
exec >>"$LOG" 2>&1

echo "[scifact-ab] started $(date '+%F %T'); waiting for the SOQA/CFMT chain"
waited=0
while pgrep -f 'run_soqa_cfmt_variants|scripts/run_coir.py' >/dev/null 2>&1; do
  waited=$((waited + 1))
  if [ "$waited" -ge 2880 ]; then
    echo "[scifact-ab] gave up after 48h"
    exit 1
  fi
  sleep 60
done
echo "[scifact-ab] chain finished; starting $(date '+%F %T')"

set -a; source /home/shawkatkabbara/memory/.env; set +a
export PAPR_API_KEY="$TEST_X_USER_API_KEY"
export PAPR_DUMP_DIR=results/scifact/vecs
PY=/home/shawkatkabbara/memory/.venv/bin/python

echo "[scifact-ab] 1/3 plain $(date '+%F %T')"
$PY scripts/run_scifact.py plain

echo "[scifact-ab] 2/3 luna (swift) BOTH sides $(date '+%F %T')"
TIER=swift EFFORT=medium $PY scripts/run_scifact.py reasoning

echo "[scifact-ab] 3/3 terra (balanced) BOTH sides $(date '+%F %T')"
TIER=balanced EFFORT=medium $PY scripts/run_scifact.py reasoning

echo "[scifact-ab] ALL DONE $(date '+%F %T')"

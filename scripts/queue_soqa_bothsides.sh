#!/usr/bin/env bash
# SOQA reasoning with the teacher on BOTH query and doc sides, queued behind
# (a) the SOQA/CFMT chain and (b) the scifact teacher A/B.
#
# WHY: the chain's SOQA reasoning stages were queries-only (teacher-predictive
# query bands vs student-literal doc bands -- a mismatched comparison that
# underprices the channel; it still showed +0.23/+0.26). This is the canonical
# both-sides shape on SOQA: 19,931 docs + 1,994 queries ~= 21.9k teacher calls
# per tier, doc extractions cached server-side per (content, schema, tier,
# effort) so re-runs only pay embedding. PAPR_DUMP_DIR persists the vectors,
# so each run yields the full band-mass surface offline (block re-weighting).
set -uo pipefail

cd "$(dirname "$0")/.."
LOG=logs/soqa_bothsides_$(date +%Y%m%d).log
mkdir -p logs
exec >>"$LOG" 2>&1

echo "[soqa-both] started $(date '+%F %T'); waiting for chain + scifact A/B"
waited=0
while pgrep -f 'run_soqa_cfmt_variants|scripts/run_coir.py|queue_scifact_teacher_ab|scripts/run_scifact.py' >/dev/null 2>&1; do
  waited=$((waited + 1))
  if [ "$waited" -ge 4320 ]; then
    echo "[soqa-both] gave up after 72h"
    exit 1
  fi
  sleep 60
done
echo "[soqa-both] queue clear; starting $(date '+%F %T')"

set -a; source /home/shawkatkabbara/memory/.env; set +a
export PAPR_API_KEY="$TEST_X_USER_API_KEY"
export PAPR_DUMP_DIR=results/coir/vecs
PY=/home/shawkatkabbara/memory/.venv/bin/python

echo "[soqa-both] 1/2 SOQA luna (swift) BOTH sides $(date '+%F %T')"
ONLY=StackOverflowQA REASONING=1 TIER=swift EFFORT=medium \
  $PY scripts/run_coir.py

echo "[soqa-both] 2/2 SOQA terra (balanced) BOTH sides $(date '+%F %T')"
ONLY=StackOverflowQA REASONING=1 TIER=balanced EFFORT=medium \
  $PY scripts/run_coir.py

echo "[soqa-both] ALL DONE $(date '+%F %T')"

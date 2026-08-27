#!/usr/bin/env bash
# Finish the SciFact 3.1.0 measurement set once the in-flight 0.6b leg clears.
#
# queue_scifact_official_frontier.sh takes a non-blocking flock and REFUSES a
# second copy rather than queuing, because both models are served by one L4 GPU
# and two concurrent runs contend instead of parallelising. So this waits for
# the process to exit rather than for the lock.
#
# Order matters. The band cache is model-agnostic (one point serves 0.6b and
# 4b for a given content/teacher config), so the 0.6b frontier/max leg already
# running is what self-heals the 980 poisoned document rows. Auditing between
# the legs is what makes the 4b numbers readable: if gaps remain, the 4b run
# would measure the same degraded vectors and we would not know it.
#
# The 4b legs use BATCH=32 to match the existing 4b 2.0.0 baselines
# (papr-embed-v1-4b-b32 = 0.78777 plain, 0.78326 frontier/max) -- the export is
# batch-invariant post e31bf066, but the batch lands in the result directory
# name, so mismatching it would scatter comparable runs across directories.
set -uo pipefail

EVALS=/home/shawkatkabbara/papr-embed-evals
MEM=/home/shawkatkabbara/memory
SCHEMA=biomedical:scifact:3.1.0
BASE_URL="${PAPR_BASE_URL:-https://memoryserver-development-7dckb3v3oa-uw.a.run.app}"
CHAIN_LOG=$EVALS/logs/chain_scifact_s31_4b.log

mkdir -p "$(dirname "$CHAIN_LOG")"
exec >>"$CHAIN_LOG" 2>&1

say() { echo "[$(date -u +%F' '%T)] $*"; }

say "chain start; base_url=$BASE_URL schema=$SCHEMA"

say "waiting for the in-flight SciFact run to exit..."
waited=0
while pgrep -f "scripts/run_scifact.py" >/dev/null; do
  sleep 60
  waited=$((waited + 1))
  if [ $((waited % 10)) -eq 0 ]; then say "  ...still running (${waited}m)"; fi
  if [ "$waited" -ge 300 ]; then say "ABORT: 5h elapsed, giving up"; exit 1; fi
done
say "0.6b leg finished after ~${waited}m"

say "auditing the ${SCHEMA} cache for surviving gaps"
"$MEM/.venv/bin/python" "$MEM/scripts/v56/audit_eval_band_coverage.py" \
  --schema "$SCHEMA" --tier frontier --effort max 2>&1 | grep -v "cache_utils\|httpx"
say "audit done (read gap_rows above; nonzero means the heal did not close)"

for VARIANT in plain reasoning; do
  say "launching 4b $VARIANT on $SCHEMA"
  PAPR_BASE_URL="$BASE_URL" \
  MODEL=papr-embed-v1-4b \
  SCHEMA_ID="$SCHEMA" \
  BATCH=32 \
    "$EVALS/scripts/queue_scifact_official_frontier.sh" "$VARIANT"
  say "4b $VARIANT rc=$?"
done

say "chain done"

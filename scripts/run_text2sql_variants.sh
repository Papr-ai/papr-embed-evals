#!/usr/bin/env bash
# Sequential text2sql variant runs (the L4 serializes inference, so parallel
# runs would slow each other and tangle the per-IP rate limit):
#
#   0. one-call frontier/max cache-hit verification (imported sol@xhigh)
#   1. swift    (gpt-5.6-luna,  medium) — reasoning on QUERIES ONLY (~$6-7)
#   2. balanced (gpt-5.6-terra, medium) — reasoning on QUERIES ONLY (~$60-70)
#   3. papr-embed-v1-4b plain            — no teacher cost
#
# Queries-only keeps teacher spend bounded by |queries| = 5,851 instead of
# |corpus| = 105,851, and matches the production shape (corpus embedded
# cheaply once, live queries carry the teacher).
set -uo pipefail

cd "$(dirname "$0")/.."
set -a; source /home/shawkatkabbara/memory/.env; set +a
export PAPR_API_KEY="$TEST_X_USER_API_KEY"
PY=/home/shawkatkabbara/memory/.venv/bin/python

echo "[chain] 0/3 frontier/max cache-hit verification $(date '+%F %T')"
QTEXT=$($PY -c "import json;print(json.loads(open('/home/shawkatkabbara/memory/training/output/data_v48_repaired/biomedical_scifact_2.0.0.jsonl').readline())['query'])")
$PY scripts/verify_frontier_cache_hit.py "$QTEXT" \
  || echo "[chain] frontier cache verification FAILED (reasoning runs unaffected; investigate before any frontier eval)"

echo "[chain] 1/3 swift (luna) queries-only $(date '+%F %T')"
ONLY=SyntheticText2SQL REASONING=1 TIER=swift EFFORT=medium QUERIES_ONLY=1 \
  $PY scripts/run_coir.py

echo "[chain] 2/3 balanced (terra) queries-only $(date '+%F %T')"
ONLY=SyntheticText2SQL REASONING=1 TIER=balanced EFFORT=medium QUERIES_ONLY=1 \
  $PY scripts/run_coir.py

echo "[chain] 3/3 papr-embed-v1-4b plain $(date '+%F %T')"
ONLY=SyntheticText2SQL MODEL=papr-embed-v1-4b \
  $PY scripts/run_coir.py

echo "[chain] ALL DONE $(date '+%F %T')"

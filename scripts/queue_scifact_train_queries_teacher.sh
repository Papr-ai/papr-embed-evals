#!/usr/bin/env bash
# Teacher-side (frontier/max) encode of the 809 SciFact TRAIN claims.
#
# COST, stated plainly: re-running a SciFact EVAL is free because the 300 test
# claims and all 5,183 documents are already in the band cache. These 809 TRAIN
# claims are not -- the cache keys on (content, schema, role, tier, effort) and
# no train claim has ever been extracted. So this bills 809 fresh gpt-5.6-sol
# xhigh extractions PER SCHEMA, ~1,618 total, about 30% of one full SciFact
# frontier/max run. It caches permanently.
#
# It does NOT bill per model. The band cache is model-agnostic, so the 0.6b leg
# of each schema pays and the 4b leg reads. That is why the order below is
# 0.6b-then-4b within each schema rather than grouped by model.
#
# Why pay at all: the plain-fitted mass does not transfer to the teacher leg.
# Borrowing plain's p_m=0.20 recovers only 16% of the teacher oracle on the
# winning 0.6b/3.1.0 leg and goes NEGATIVE on 0.6b/2.0.0, because teacher bands
# are better and therefore want more mass (oracle 0.31, not 0.20). Fitting the
# teacher leg on its own train claims is the only honest way to get its number.
set -uo pipefail

EVALS=/home/shawkatkabbara/papr-embed-evals
MEM=/home/shawkatkabbara/memory
LOG=$EVALS/logs/scifact_train_queries_teacher.log

mkdir -p "$(dirname "$LOG")"
cd "$EVALS" || exit 1

set -a
# shellcheck disable=SC1091
source "$MEM/.env"
set +a

export PAPR_API_KEY="${TEST_X_USER_API_KEY:?TEST_X_USER_API_KEY missing from $MEM/.env}"
export PAPR_BASE_URL="${PAPR_BASE_URL:-https://memory.papr.ai}"
export PYTHONUNBUFFERED=1
export TIER=frontier EFFORT=max
export PAPR_HTTP_TIMEOUT_S="${PAPR_HTTP_TIMEOUT_S:-650}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

exec >>"$LOG" 2>&1
echo "=================================================================="
echo "[$(date -u +%F' '%T)] scifact TRAIN teacher encode start pid=$$ base=$PAPR_BASE_URL"

run() {
  local model=$1 schema=$2 batch=$3 conc=$4 slug=$5
  echo "--- [$(date -u +%T)] $model schema=${schema:-2.0.0-default} batch=$batch conc=$conc"
  MODEL="$model" SCHEMA_ID="$schema" BATCH="$batch" PAPR_EMBED_CONCURRENCY="$conc" \
  PAPR_DUMP_DIR="$EVALS/results/scifact/vecs_${slug}_frontier_max" \
    "$MEM/.venv/bin/python" -u scripts/embed_scifact_train_queries.py
  echo "    rc=$?"
}

# 0.6b first in each schema: it pays the extraction, the 4b then reads cache.
# batch 8 on the 0.6b matches PAPR_TEACHER_CONCURRENCY so one request is one
# extraction wave; the 4b uses batch 32 only because its dumps are named -b32
# and the glob has to match, with concurrency dropped to 3 after the plain run
# took 14 retries on 503s at concurrency 6.
run papr-embed-v1-0.6b ""                        8  6 0p6b
run papr-embed-v1-4b   ""                        32 3 4b
run papr-embed-v1-0.6b biomedical:scifact:3.1.0  8  6 0p6b_s3p1p0
run papr-embed-v1-4b   biomedical:scifact:3.1.0  32 3 4b_s3p1p0

echo "[$(date -u +%F' '%T)] done"

#!/usr/bin/env bash
# Encode the 809 SciFact TRAIN claims for both towers, plain path only.
#
# Plain means no teacher extraction, so this costs GPU encode time and nothing
# else -- there is no gpt-5.6-sol call anywhere in it. The 5,183 document
# vectors are already dumped by the ordinary test runs and the corpus is shared
# across splits, so these 809 query vectors are the only missing piece needed
# to fit band mass off-test.
#
# Dumps land in the SAME directory as the matching test-split run, because the
# filename carries SciFactTrain and the sweep globs on that. Keeping them
# together means a train fit and its test readout can never drift onto
# different document vectors.
set -uo pipefail

EVALS=/home/shawkatkabbara/papr-embed-evals
MEM=/home/shawkatkabbara/memory
LOG=$EVALS/logs/scifact_train_queries.log

mkdir -p "$(dirname "$LOG")"
cd "$EVALS" || exit 1

set -a
# shellcheck disable=SC1091
source "$MEM/.env"
set +a

export PAPR_API_KEY="${TEST_X_USER_API_KEY:?TEST_X_USER_API_KEY missing from $MEM/.env}"
export PAPR_BASE_URL="${PAPR_BASE_URL:-https://memory.papr.ai}"
export PYTHONUNBUFFERED=1
export PAPR_EMBED_CONCURRENCY="${PAPR_EMBED_CONCURRENCY:-6}"
export PAPR_HTTP_TIMEOUT_S="${PAPR_HTTP_TIMEOUT_S:-650}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

exec >>"$LOG" 2>&1
echo "=================================================================="
echo "[$(date -u +%F' '%T)] scifact TRAIN query encode start pid=$$ base=$PAPR_BASE_URL"

# Both towers, both schemas. Batch matches the test-split run for each tower so
# the dump names line up with the existing baselines.
run() {
  local model=$1 schema=$2 batch=$3 slug=$4
  echo "--- $model schema=$schema batch=$batch -> vecs_${slug}"
  MODEL="$model" SCHEMA_ID="$schema" BATCH="$batch" \
  PAPR_DUMP_DIR="$EVALS/results/scifact/vecs_${slug}_frontier_max" \
    "$MEM/.venv/bin/python" -u scripts/embed_scifact_train_queries.py
  echo "    rc=$?"
}

run papr-embed-v1-0.6b ""                          8  0p6b
run papr-embed-v1-0.6b biomedical:scifact:3.1.0    8  0p6b_s3p1p0
run papr-embed-v1-4b   ""                          32 4b
run papr-embed-v1-4b   biomedical:scifact:3.1.0    32 4b_s3p1p0

echo "[$(date -u +%F' '%T)] done"

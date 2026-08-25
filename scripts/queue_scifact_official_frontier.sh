#!/usr/bin/env bash
# Official MTEB SciFact (300 test claims x 5,183 abstracts) through the public
# production API, papr-embed-v1-0.6b + gpt-5.6-sol reasoning on BOTH sides.
#
# This runs the real mteb harness (mteb.MTEB(...).run), so it emits standard
# MTEB result JSON under results/scifact/<model_name>/ -- the artifact a
# leaderboard submission consumes. Nothing about the scoring path is bespoke.
#
# frontier/max resolves server-side to gpt-5.6-sol @ xhigh, which is the effort
# the shipped teacher corpora were built at. Documents are extracted literally,
# queries through the predictive frame -- variant D, the measured best teacher
# program for scifact (docs/training/V55A_TASK_CONDITIONED_TEACHER_PROMPTS.md
# section 13.1: mu +0.0048, the only positive column besides C').
#
# Every extraction is cached server-side in Qdrant papr_band_cache_v1, keyed by
# (content, schema, role, tier, effort), so this is resumable and every later
# run over the same corpus is free. A 503 (the fail-closed teacher error) is
# retried with backoff by the client, and the retry only re-extracts whatever
# did not already land in the cache.
#
#   ./scripts/queue_scifact_official_frontier.sh            # plain + reasoning
#   ./scripts/queue_scifact_official_frontier.sh reasoning  # teacher only
#
# PRECONDITION: scripts/v58/verify_teacher_fix_live.py must PASS against the
# deployed revision. Without it you cannot tell a working teacher from one
# silently caching 14 blanks.
set -uo pipefail

VARIANT="${1:-both}"
EVALS=/home/shawkatkabbara/papr-embed-evals
MEM=/home/shawkatkabbara/memory

# Resolve MODEL before the paths: the dump feeds the offline (p_m, p_e) mass
# surface, so a 4b run writing to the 0.6b directory would silently destroy the
# baseline it is meant to be compared against. papr-embed-v1-0.6b -> 0p6b keeps
# the existing 0.6b dump path unchanged.
MODEL="${MODEL:-papr-embed-v1-0.6b}"
SLUG=$(printf '%s' "$MODEL" | sed 's/^papr-embed-v1-//; s/\./p/g')
LOG=$EVALS/logs/scifact_official_frontier_max_${SLUG}.log
DUMP=$EVALS/results/scifact/vecs_${SLUG}_frontier_max
# Deliberately NOT per-model: both models are served by the same single L4 GPU,
# so concurrent runs would contend rather than parallelise.
LOCK=$EVALS/logs/scifact_official_frontier_max.lock

mkdir -p "$(dirname "$LOG")" "$DUMP" "$EVALS/results/scifact"
cd "$EVALS" || exit 1

# One run at a time: two copies would double-bill every cache miss that is
# in flight in both (the cache only dedupes what has already been written).
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "official SciFact run already holds $LOCK; refusing a second copy" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source "$MEM/.env"
set +a

export PAPR_API_KEY="${TEST_X_USER_API_KEY:?TEST_X_USER_API_KEY missing from $MEM/.env}"
export PAPR_BASE_URL="${PAPR_BASE_URL:-https://memory.papr.ai}"
export PYTHONUNBUFFERED=1

export MODEL
export TIER="${TIER:-frontier}"
export EFFORT="${EFFORT:-max}"
export QUERIES_ONLY="${QUERIES_ONLY:-0}"   # 0 = teacher on BOTH sides

# BATCH 8 matches the server's per-request teacher fan-out
# (PAPR_TEACHER_CONCURRENCY=8), so a request is ONE extraction wave and stays
# far under Cloud Run's 600s request timeout. Throughput comes from running
# several such requests at once instead of from a bigger batch.
export BATCH="${BATCH:-8}"
export PAPR_EMBED_CONCURRENCY="${PAPR_EMBED_CONCURRENCY:-6}"
export PAPR_HTTP_TIMEOUT_S="${PAPR_HTTP_TIMEOUT_S:-650}"

# One dumped run yields the whole (p_m, p_e) band-mass surface offline, so the
# mass calibration never needs a second paid pass.
export PAPR_DUMP_DIR="$DUMP"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec >>"$LOG" 2>&1
echo "=================================================================="
echo "[$(date -u +%F' '%T)] official SciFact start pid=$$ variant=$VARIANT"
echo "  model=$MODEL reasoning=$TIER/$EFFORT both_sides=$((1 - QUERIES_ONLY))"
echo "  batch=$BATCH concurrency=$PAPR_EMBED_CONCURRENCY timeout=${PAPR_HTTP_TIMEOUT_S}s"
set +e
"$MEM/.venv/bin/python" -u scripts/run_scifact.py "$VARIANT"
rc=$?
echo "[$(date -u +%F' '%T)] done rc=$rc"
exit "$rc"

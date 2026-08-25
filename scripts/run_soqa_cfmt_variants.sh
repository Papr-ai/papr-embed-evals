#!/usr/bin/env bash
# StackOverflowQA + CodeFeedbackMT variant matrix through the production API.
#
# Task rationale (V55A_COIR_INSTRUCTION_PARITY_EVAL.md): SOQA carries a real
# -0.89 deficit vs stock on a non-degenerate corpus (19,931 docs / 1,994
# queries -- cheapest full matrix). CFMT carries the only attributed
# model-side deficit in CoIR (-0.7) and its multi-turn queries are 8.6%
# truncated at 2048, the one case where a query-side teacher can carry
# information the plain student physically cannot see.
#
# Order: cheapest signal first. 4b stages run with PAPR_EMBED_CONCURRENCY=4
# (the hosted 4B endpoint has multiple replicas; one serial stream leaves
# them idle -- text2sql 4b took 10h serial). 0.6b stages stay serial: at
# ~2.5s/request, 4 streams would brush the 100 req/min per-IP rate limit.
# CFMT terra (~13.3k queries, ~$150) is intentionally NOT here -- gate it on
# CFMT luna showing any lift.
set -uo pipefail

cd "$(dirname "$0")/.."
set -a; source /home/shawkatkabbara/memory/.env; set +a
export PAPR_API_KEY="$TEST_X_USER_API_KEY"
PY=/home/shawkatkabbara/memory/.venv/bin/python

echo "[chain] 1/7 SOQA 0.6b plain $(date '+%F %T')"
ONLY=StackOverflowQA $PY scripts/run_coir.py

echo "[chain] 2/7 SOQA swift (luna) queries-only $(date '+%F %T')"
ONLY=StackOverflowQA REASONING=1 TIER=swift EFFORT=medium QUERIES_ONLY=1 \
  $PY scripts/run_coir.py

echo "[chain] 3/7 SOQA balanced (terra) queries-only $(date '+%F %T')"
ONLY=StackOverflowQA REASONING=1 TIER=balanced EFFORT=medium QUERIES_ONLY=1 \
  $PY scripts/run_coir.py

echo "[chain] 4/7 SOQA 4b plain conc=4 $(date '+%F %T')"
ONLY=StackOverflowQA MODEL=papr-embed-v1-4b PAPR_EMBED_CONCURRENCY=4 \
  $PY scripts/run_coir.py

echo "[chain] 5/7 CFMT 0.6b plain $(date '+%F %T')"
ONLY=CodeFeedbackMT $PY scripts/run_coir.py

echo "[chain] 6/7 CFMT swift (luna) queries-only $(date '+%F %T')"
ONLY=CodeFeedbackMT REASONING=1 TIER=swift EFFORT=medium QUERIES_ONLY=1 \
  $PY scripts/run_coir.py

echo "[chain] 7/7 CFMT 4b plain conc=4 $(date '+%F %T')"
ONLY=CodeFeedbackMT MODEL=papr-embed-v1-4b PAPR_EMBED_CONCURRENCY=4 \
  $PY scripts/run_coir.py

echo "[chain] ALL DONE $(date '+%F %T')"

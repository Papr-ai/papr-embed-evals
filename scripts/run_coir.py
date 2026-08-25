#!/usr/bin/env python3
"""CoIR suite through the public API, cheapest task first.

The two CodeSearchNet tasks have 280k-1M document corpora; through a
64-text-per-request API that is thousands of requests, so they sit last and
the runner prints a cost estimate before starting each task.

    PAPR_API_KEY=... python scripts/run_coir.py                        # full suite
    PAPR_API_KEY=... ONLY="CosQA CodeTransOceanDL" python scripts/run_coir.py
    PAPR_API_KEY=... REASONING=1 python scripts/run_coir.py            # reasoning variant

Variant knobs (all optional):
    MODEL=papr-embed-v1-4b        override the per-task default model
    REASONING=1 TIER=frontier EFFORT=max     reasoning tier/effort
    QUERIES_ONLY=1                COST-BOUNDING EXCEPTION ONLY: reasoning on
                                  queries while documents keep student bands --
                                  a mismatched band comparison that underprices
                                  the teacher. The default (unset) hits BOTH
                                  sides, which is the canonical reasoning shape.
    BATCH=32                      texts per API request (default 64; the model
                                  is batch-invariant post-e31bf066, so this is
                                  a noise-floor probe, not a quality knob).
                                  Non-default values get a -b{N} results suffix.
    PAPR_EMBED_CONCURRENCY=4      requests kept in flight (default 1). Use on
                                  4b runs (multi-replica hosted base tower);
                                  keep 0.6b serial (per-IP rate limit).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

# Progress visibility for nohup/background runs: the API wrapper logs one
# line every ~6400 inputs; without this the log is silent until the score.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
)

sys.path.insert(0, ".")

import mteb  # noqa: E402

from papr_embed_evals.client import ReasoningOptions  # noqa: E402
from papr_embed_evals.mteb_model import PaprEmbedAPIModel  # noqa: E402
from papr_embed_evals.tasks import COIR_ORDER, TASKS  # noqa: E402

RESULTS_DIR = Path("results/coir")


def main() -> None:
    only = set(os.getenv("ONLY", "").split()) or None
    model_override = os.getenv("MODEL") or None
    reasoning = (
        ReasoningOptions(
            tier=os.getenv("TIER", "swift"),        # type: ignore[arg-type]
            effort=os.getenv("EFFORT", "medium"),   # type: ignore[arg-type]
        )
        if os.getenv("REASONING", "0") == "1"
        else None
    )
    queries_only = os.getenv("QUERIES_ONLY", "0") == "1"
    batch = int(os.getenv("BATCH", "64"))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for task_name in COIR_ORDER:
        if only and task_name not in only:
            continue
        spec = TASKS[task_name]
        est_requests = (spec.corpus_size or 0) // 64
        print(
            f"--- {task_name}: schema={spec.schema_id} "
            f"corpus~{spec.corpus_size} (~{est_requests} document requests)"
        )

        model = PaprEmbedAPIModel(
            task_name,
            model_id=model_override,
            reasoning=reasoning,
            reasoning_queries_only=queries_only,
        )
        if batch != 64:
            # Separate results/summary from the batch-64 runs of the same model.
            model.model_name = f"{model.model_name}-b{batch}"
        tasks = mteb.get_tasks(tasks=[spec.mteb_task])
        evaluator = mteb.MTEB(tasks=tasks)

        started = time.time()
        try:
            results = evaluator.run(
                model,
                output_folder=str(RESULTS_DIR / task_name / model.model_name),
                overwrite_results=True,
                encode_kwargs={"batch_size": batch},
            )
            scores = results[0].scores["test"][0]
            row = {
                "task": task_name,
                "model": model.model_name,
                "ndcg_at_10": scores["ndcg_at_10"],
                "minutes": round((time.time() - started) / 60, 1),
                "api_stats": model.client.stats.as_dict(),
            }
        except Exception as exc:
            row = {"task": task_name, "model": model.model_name, "error": str(exc)[:500]}
        print(json.dumps(row, indent=2))
        rows.append(row)
        summary_path = RESULTS_DIR / f"summary_{model.model_name}.json"
        summary_path.write_text(json.dumps(rows, indent=2))

    finished = [r for r in rows if "ndcg_at_10" in r]
    if finished:
        mean = sum(r["ndcg_at_10"] for r in finished) / len(finished)
        print(f"\nmean NDCG@10 over {len(finished)} tasks: {mean:.5f}")
    if rows:
        print(f"written -> {RESULTS_DIR}/summary_*.json")


if __name__ == "__main__":
    main()

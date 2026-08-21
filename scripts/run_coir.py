#!/usr/bin/env python3
"""CoIR suite through the public API, cheapest task first.

The two CodeSearchNet tasks have 280k-1M document corpora; through a
64-text-per-request API that is thousands of requests, so they sit last and
the runner prints a cost estimate before starting each task.

    PAPR_API_KEY=... python scripts/run_coir.py                        # full suite
    PAPR_API_KEY=... ONLY="CosQA CodeTransOceanDL" python scripts/run_coir.py
    PAPR_API_KEY=... REASONING=1 python scripts/run_coir.py            # reasoning variant
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import mteb  # noqa: E402

from papr_embed_evals.client import ReasoningOptions  # noqa: E402
from papr_embed_evals.mteb_model import PaprEmbedAPIModel  # noqa: E402
from papr_embed_evals.tasks import COIR_ORDER, TASKS  # noqa: E402

RESULTS_DIR = Path("results/coir")


def main() -> None:
    only = set(os.getenv("ONLY", "").split()) or None
    reasoning = (
        ReasoningOptions(tier="swift", effort="medium")
        if os.getenv("REASONING", "0") == "1"
        else None
    )

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

        model = PaprEmbedAPIModel(task_name, reasoning=reasoning)
        tasks = mteb.get_tasks(tasks=[spec.mteb_task])
        evaluator = mteb.MTEB(tasks=tasks)

        started = time.time()
        try:
            results = evaluator.run(
                model,
                output_folder=str(RESULTS_DIR / task_name),
                overwrite_results=True,
                encode_kwargs={"batch_size": 64},
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
            row = {"task": task_name, "error": str(exc)[:500]}
        print(json.dumps(row, indent=2))
        rows.append(row)
        (RESULTS_DIR / "summary.json").write_text(json.dumps(rows, indent=2))

    finished = [r for r in rows if "ndcg_at_10" in r]
    if finished:
        mean = sum(r["ndcg_at_10"] for r in finished) / len(finished)
        print(f"\nmean NDCG@10 over {len(finished)} tasks: {mean:.5f}")
    print(f"written -> {RESULTS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""SciFact NDCG@10 through the public API: plain vs reasoning-boosted.

SciFact is the cheapest full-consistency reasoning testbed: 300 test queries,
5,183 documents, so a BOTH-SIDES teacher run (documents extracted literally at
index time, queries predictively at search time -- the configuration whose
band cosine compares like with like) costs ~5.5k teacher calls. Teacher
extractions are cached server-side per (content, schema, tier, effort), so
re-runs only pay for embedding.

    PAPR_API_KEY=... python scripts/run_scifact.py                 # both variants
    PAPR_API_KEY=... python scripts/run_scifact.py plain           # one variant
    PAPR_API_KEY=... python scripts/run_scifact.py reasoning

Variant knobs (all optional):
    MODEL=papr-embed-v1-4b        override the default pinned-schema 0.6b model
    TIER=swift EFFORT=medium      reasoning tier/effort (default swift/medium)
    QUERIES_ONLY=1                restrict the teacher to queries (the cheap
                                  asymmetric shape; default is BOTH sides)
    BATCH=32                      texts per API request (default 64)
    PAPR_DUMP_DIR=results/scifact/vecs   persist returned vectors; one dumped
                                  run gives the whole band-mass surface offline
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import mteb  # noqa: E402

from papr_embed_evals.client import ReasoningOptions  # noqa: E402
from papr_embed_evals.mteb_model import PaprEmbedAPIModel  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(message)s"
)

RESULTS_DIR = Path("results/scifact")


def run_variant(variant: str) -> dict:
    tier = os.getenv("TIER", "swift")
    effort = os.getenv("EFFORT", "medium")
    queries_only = os.getenv("QUERIES_ONLY", "0") == "1"
    batch = int(os.getenv("BATCH", "64"))
    reasoning = (
        ReasoningOptions(tier=tier, effort=effort)
        if variant == "reasoning"
        else None
    )
    model = PaprEmbedAPIModel(
        "SciFact",
        model_id=os.getenv("MODEL") or None,
        reasoning=reasoning,
        reasoning_queries_only=queries_only,
    )
    if batch != 64:
        model.model_name = f"{model.model_name}-b{batch}"

    tasks = mteb.get_tasks(tasks=["SciFact"])
    evaluator = mteb.MTEB(tasks=tasks)

    out = RESULTS_DIR / model.model_name
    started = time.time()
    results = evaluator.run(
        model,
        output_folder=str(out),
        overwrite_results=True,
        encode_kwargs={"batch_size": batch},
    )
    minutes = (time.time() - started) / 60

    scores = results[0].scores["test"][0]
    summary = {
        "variant": variant,
        "model": model.model_name,
        "ndcg_at_10": scores["ndcg_at_10"],
        "minutes": round(minutes, 1),
        "api_stats": model.client.stats.as_dict(),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    variants = ["plain", "reasoning"] if which == "both" else [which]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summaries = [run_variant(v) for v in variants]

    stamp = int(time.time())
    (RESULTS_DIR / f"summary_{stamp}.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nwritten -> {RESULTS_DIR / f'summary_{stamp}.json'}")
    if len(summaries) == 2:
        delta = summaries[1]["ndcg_at_10"] - summaries[0]["ndcg_at_10"]
        print(f"reasoning lift on SciFact NDCG@10: {delta:+.5f}")


if __name__ == "__main__":
    main()

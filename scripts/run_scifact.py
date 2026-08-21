#!/usr/bin/env python3
"""SciFact NDCG@10 through the public API: plain vs reasoning-boosted.

This is the first eval this repo runs (SciFact is small: 300 test queries,
5,183 documents — roughly 90 API batches per variant, cheap enough to run
both variants in one sitting; the reasoning variant's teacher extractions
are cached server-side, so re-runs only pay for embedding).

    PAPR_API_KEY=... python scripts/run_scifact.py                 # both variants
    PAPR_API_KEY=... python scripts/run_scifact.py plain           # one variant
    PAPR_API_KEY=... python scripts/run_scifact.py reasoning
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import mteb  # noqa: E402

from papr_embed_evals.client import ReasoningOptions  # noqa: E402
from papr_embed_evals.mteb_model import PaprEmbedAPIModel  # noqa: E402

RESULTS_DIR = Path("results/scifact")


def run_variant(variant: str) -> dict:
    reasoning = ReasoningOptions(tier="swift", effort="medium") if variant == "reasoning" else None
    model = PaprEmbedAPIModel("SciFact", reasoning=reasoning)

    tasks = mteb.get_tasks(tasks=["SciFact"])
    evaluator = mteb.MTEB(tasks=tasks)

    out = RESULTS_DIR / variant
    started = time.time()
    results = evaluator.run(
        model,
        output_folder=str(out),
        overwrite_results=True,
        encode_kwargs={"batch_size": 64},
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

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nwritten -> {RESULTS_DIR / 'summary.json'}")
    if len(summaries) == 2:
        delta = summaries[1]["ndcg_at_10"] - summaries[0]["ndcg_at_10"]
        print(f"reasoning lift on SciFact NDCG@10: {delta:+.5f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pre-flight: prove the API key, models, and reasoning path work end to end.

Run this before any eval. It costs a handful of embedding calls.

    PAPR_API_KEY=... python scripts/smoke_api.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from papr_embed_evals.client import PaprEmbeddingsClient, ReasoningOptions  # noqa: E402


def main() -> int:
    client = PaprEmbeddingsClient()
    print(f"endpoint: {client.base_url}")

    models = client.list_models()
    live = [m["id"] for m in models if m["status"] == "live"]
    print(f"models ({len(models)} listed, {len(live)} live): {live}")

    query = "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?"
    doc = (
        "Programmed cell death (PCD) is the regulated death of cells within an organism. "
        "The lace plant produces perforations in its leaves through PCD; mitochondrial "
        "dynamics were examined during this process."
    )

    checks = [
        ("scifact / plain", "papr-embed-v1-0.6b-scifact", None),
        ("scifact / reasoning", "papr-embed-v1-0.6b-scifact", ReasoningOptions()),
    ]
    failures = 0
    for label, model_id, reasoning in checks:
        try:
            q = client.embed([query], model=model_id, input_type="query", reasoning=reasoning)
            d = client.embed([doc], model=model_id, input_type="document", reasoning=reasoning)
            cos = float((q[0] / (q[0] ** 2).sum() ** 0.5) @ (d[0] / (d[0] ** 2).sum() ** 0.5))
            print(f"  ok  {label}: dim={q.shape[1]}, cos(q,d)={cos:.4f}")
        except Exception as exc:  # pragma: no cover - smoke output
            failures += 1
            print(f"  FAIL {label}: {exc}")

    print(f"stats: {client.stats.as_dict()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

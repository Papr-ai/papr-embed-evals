#!/usr/bin/env python3
"""Zero-cost proof that the imported sol@xhigh cache serves frontier/max.

Embeds ONE scifact training query (present in the imported band cache) with
reasoning tier=frontier effort=max and asserts the response reports a cache
hit and zero uncached teacher runs. If this reports uncached=1 instead, the
import keying is wrong and the call just paid for one sol@max extraction
(~$0.10-0.25) -- the cheapest possible way to find that out.

    PAPR_API_KEY=... python scripts/verify_frontier_cache_hit.py
"""

from __future__ import annotations

import json
import sys

import httpx

sys.path.insert(0, ".")

from papr_embed_evals.client import PaprEmbeddingsClient  # noqa: E402

# First query of training/output/data_v48_repaired/biomedical_scifact_2.0.0.jsonl,
# passed on the command line by the caller to avoid baking dataset text here.
TEXT = sys.argv[1] if len(sys.argv) > 1 else None
if not TEXT:
    print("usage: verify_frontier_cache_hit.py '<scifact training query text>'")
    raise SystemExit(2)

client = PaprEmbeddingsClient()
response = client._request(
    "POST",
    "/v1/embeddings",
    json={
        "model": "papr-embed-v1-0.6b-scifact",
        "input": [TEXT],
        "input_type": "query",
        "reasoning": {"tier": "frontier", "effort": "max"},
    },
)
body = response.json()
meta = body.get("meta", {})
reasoning = meta.get("reasoning") or {}
print(json.dumps({
    "dim": len(body["data"][0]["embedding"]),
    "latency_ms": meta.get("latency_ms"),
    "reasoning": reasoning,
}, indent=1))

hits = int(reasoning.get("cache_hits") or 0)
uncached = int(reasoning.get("uncached") or 0)
if hits == 1 and uncached == 0:
    print("VERIFIED: frontier/max served from the imported sol@xhigh cache ($0 LLM spend).")
    raise SystemExit(0)
print("CACHE MISS: the serving key did not match the imported entry -- investigate before any frontier run.")
raise SystemExit(1)

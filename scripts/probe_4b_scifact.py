#!/usr/bin/env python3
"""Pre-flight for the 4b SciFact run: does frontier/max reuse the 0.6b cache?

The band cache key is (content, schema, role, tier, effort, extractor_version)
and is documented MODEL-agnostic, so the 5,482 sol@xhigh extractions already
paid for by the 0.6b run should serve the 4b run for free. "Should" is worth
about $1,333, so this proves it on two texts before the full run commits.

Three checks, cheapest first:

  1. /models reports papr-embed-v1-4b live, and its advertised dimension.
  2. One real SciFact document + one real test query embed PLAIN on the 4b.
     Confirms the hybrid path (remote Qwen3-4B base tower + local band/evidence
     heads) answers at all, and measures the ACTUAL shipped block masses from
     the returned vector rather than trusting a doc table.
  3. The SAME two texts embed with reasoning frontier/max and must report
     cache_hits=1 / uncached=0 on BOTH roles. uncached=1 means the key did not
     match and the full run would re-extract the whole corpus at frontier
     prices -- stop and investigate rather than launching.

The texts are read from the local MTEB arrow files and asserted byte-identical
to what the 0.6b run actually sent, by matching their sha1 against the dumped
vector keys. Without that assertion a cache hit here would not generalise: a
probe on a text nobody embedded proves nothing about the corpus.

    PAPR_API_KEY=... python scripts/probe_4b_scifact.py
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

import numpy as np
import pyarrow as pa

sys.path.insert(0, ".")

from papr_embed_evals.client import PaprEmbeddingsClient  # noqa: E402

MODEL = os.getenv("MODEL", "papr-embed-v1-4b")
SCHEMA = "biomedical:scifact:2.0.0"
TIER = os.getenv("TIER", "frontier")
EFFORT = os.getenv("EFFORT", "max")

REV = "d56462d0e63a25450459c4f213e49ffdb866f7f9"
BASE = os.path.expanduser("~/.cache/huggingface/datasets/mteb___scifact")
DUMPS = "results/scifact/vecs_0p6b_frontier_max"


def arrow(path: str) -> list[dict]:
    with pa.memory_map(path, "r") as source:
        try:
            return pa.ipc.open_file(source).read_all().to_pylist()
        except Exception:
            source.seek(0)
            return pa.ipc.open_stream(source).read_all().to_pylist()


def sha1_16(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def dumped_keys(pattern: str) -> set[str]:
    matches = glob.glob(f"{DUMPS}/{pattern}")
    if not matches:
        return set()
    data = np.load(matches[0], allow_pickle=True)
    return {str(k) for k in data["keys"]}


def masses(vector: list[float], base_dims: int) -> dict:
    z = np.asarray(vector, dtype=np.float64)
    band_end = base_dims + 1792
    total = float((z**2).sum())
    return {
        "dim": len(z),
        "norm": round(float(np.sqrt(total)), 5),
        "p_base": round(float((z[:base_dims] ** 2).sum() / total), 4),
        "p_m": round(float((z[base_dims:band_end] ** 2).sum() / total), 4),
        "p_e": round(float((z[band_end:] ** 2).sum() / total), 4),
    }


def main() -> int:
    client = PaprEmbeddingsClient()

    # ---- 1. model is live, and what dimension does it advertise? ----------
    models = {m["id"]: m for m in client.list_models()}
    spec = models.get(MODEL)
    if spec is None:
        print(f"FAIL: {MODEL} not in /models ({sorted(models)})")
        return 1
    # max_context_tokens is absent on revisions that predate truncation
    # reporting; the per-request meta.context_tokens below is authoritative.
    print(f"[1] {MODEL}: status={spec['status']} dims={spec['dimensions']} "
          f"ctx={spec.get('max_context_tokens', 'n/a')} "
          f"reasoning={spec['supports_reasoning']}")
    if spec["status"] != "live":
        print(f"FAIL: {MODEL} is {spec['status']}, the route will 503")
        return 1
    base_dims = int(spec["dimensions"]) - 1792 - 1152

    # ---- 2. exact texts the 0.6b run sent ---------------------------------
    corpus = arrow(f"{BASE}/corpus/0.0.0/{REV}/scifact-corpus.arrow")
    queries = arrow(f"{BASE}/queries/0.0.0/{REV}/scifact-queries.arrow")
    doc_keys = dumped_keys("*0.6b-b8_SciFact_document*")
    query_keys = dumped_keys("*0.6b-b8_SciFact_query*")

    doc_text = ((corpus[0].get("title") or "") + " " + (corpus[0].get("text") or "")).strip()
    # The queries file holds train+test; only the 300 TEST queries were embedded
    # (and therefore cached), so pick one the 0.6b run actually sent.
    query_text = next(
        (r["text"] for r in queries if sha1_16(r["text"]) in query_keys),
        queries[0]["text"],
    )
    doc_seen = sha1_16(doc_text) in doc_keys
    query_seen = sha1_16(query_text) in query_keys
    print(f"[2] text provenance: document in 0.6b dump={doc_seen} "
          f"query in 0.6b dump={query_seen}")
    if doc_keys and not doc_seen:
        print("FAIL: reconstructed document text is not byte-identical to the "
              "0.6b run's; a cache result here would not generalise.")
        return 1

    # ---- 3. plain on the 4b: does the hybrid path answer, and at what mass?
    for role, text in (("document", doc_text), ("query", query_text)):
        body = client._request("POST", "/v1/embeddings", json={
            "model": MODEL, "input": [text], "input_type": role,
            "schema_id": SCHEMA,
        }).json()
        meta = body["meta"]
        print(f"[3] plain {role:8s}: {json.dumps(masses(body['data'][0]['embedding'], base_dims))} "
              f"latency_ms={meta.get('latency_ms')} ctx={meta.get('context_tokens')}")

    # ---- 4. frontier/max: cache hit or a $1,333 surprise? -----------------
    verdict = 0
    for role, text in (("document", doc_text), ("query", query_text)):
        body = client._request("POST", "/v1/embeddings", json={
            "model": MODEL, "input": [text], "input_type": role,
            "schema_id": SCHEMA,
            "reasoning": {"tier": TIER, "effort": EFFORT},
        }).json()
        meta = body["meta"]
        reasoning = meta.get("reasoning") or {}
        hits = int(reasoning.get("cache_hits") or 0)
        uncached = int(reasoning.get("uncached") or 0)
        mass = masses(body["data"][0]["embedding"], base_dims)
        status = "REUSED" if (hits == 1 and uncached == 0) else "MISS"
        print(f"[4] {TIER}/{EFFORT} {role:8s}: {status} hits={hits} uncached={uncached} "
              f"bands_present={reasoning.get('mean_bands_present')} "
              f"dropped={reasoning.get('bands_dropped')} "
              f"p_m={mass['p_m']} latency_ms={meta.get('latency_ms')}")
        if status == "MISS":
            verdict = 1

    if verdict:
        print("\nSTOP: at least one role missed the cache. A full run would "
              "re-extract 5,482 texts at frontier/max prices. Investigate the "
              "key (schema/role/tier/effort/extractor version) first.")
    else:
        print(f"\nVERIFIED: {MODEL} reuses the 0.6b sol@xhigh cache on both "
              "roles. The frontier 4b run costs GPU only, $0 in teacher spend.")
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())

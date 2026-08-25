#!/usr/bin/env python3
"""Is the 4b hybrid path slow, cold, or broken? One call each, no retries.

The 4b's base block comes from a hosted multi-replica Qwen3-4B tower, so a
first call after idle can pay a cold-start that looks identical to a hang from
the outside. The eval client retries with backoff, which buries the actual
status code. This makes ONE unretried call per model with an explicit timeout
and prints what came back, so a cold start (slow 200) can be told apart from a
dead upstream (503) or a timeout.

The 0.6b runs first as the control: it shares the whole request path except the
remote base tower, so if 0.6b is fast and 4b is not, the tower is the problem.

    PAPR_API_KEY=... python scripts/diag_4b_latency.py [timeout_s] [repeats]
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, ".")

from papr_embed_evals.client import PaprEmbeddingsClient  # noqa: E402

TIMEOUT = float(sys.argv[1]) if len(sys.argv) > 1 else 240.0
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
SCHEMA = "biomedical:scifact:2.0.0"
TEXT = "Short control text about virus replication in host cells."


def main() -> int:
    # max_attempts=1: a retry would hide the first response, which is the one
    # that distinguishes a cold start from a failure.
    client = PaprEmbeddingsClient(timeout_s=TIMEOUT, max_attempts=1)
    worst = 0
    for model in ("papr-embed-v1-0.6b", "papr-embed-v1-4b"):
        for attempt in range(1, REPEATS + 1):
            started = time.time()
            label = f"{model:20s} call {attempt}"
            try:
                response = client._request("POST", "/v1/embeddings", json={
                    "model": model,
                    "input": [TEXT],
                    "input_type": "document",
                    "schema_id": SCHEMA,
                })
                elapsed = time.time() - started
                body = response.json()
                print(f"{label}: 200 in {elapsed:6.1f}s "
                      f"dim={len(body['data'][0]['embedding'])} "
                      f"meta={json.dumps(body.get('meta', {}))}")
            except Exception as exc:
                elapsed = time.time() - started
                # PaprAPIError carries the status; anything else is transport.
                print(f"{label}: {type(exc).__name__} after {elapsed:6.1f}s: "
                      f"{str(exc)[:240]}")
                worst = 1
    return worst


if __name__ == "__main__":
    raise SystemExit(main())

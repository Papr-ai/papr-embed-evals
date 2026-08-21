"""HTTP client for the Papr embeddings API.

This is a thin, dependency-light client used until the official Python SDK
(``papr-memory``) exposes the embeddings route; the public surface here
(``embed(texts, model=..., input_type=..., schema_id=..., reasoning=...)``)
is deliberately shaped so swapping the transport for the SDK later is a
one-file change.

Auth: set ``PAPR_API_KEY`` in the environment (sent as ``X-API-Key``).
Endpoint: ``PAPR_BASE_URL`` (default ``https://memory.papr.ai``).

Server-side facts this client encodes:
  - max 64 texts per request, 100k combined characters per request
  - queries receive the schema-resolved task instruction ON THE SERVER
    (the checkpoint bakes the recipe), so callers must send RAW text and
    never prepend "Instruct: ..." themselves
  - ``reasoning`` triggers live-teacher band extraction; extractions are
    cached server-side per (content, schema, role), so re-runs over the
    same corpus only pay for cache misses
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional, Sequence

import httpx
import numpy as np

MAX_BATCH_SIZE = 64
MAX_INPUT_CHARS = 100_000
# Leave headroom under the server's hard limit so a batch never 422s.
CHAR_BUDGET = 90_000

InputType = Literal["query", "document"]


class PaprAPIError(RuntimeError):
    """Raised when the API returns a non-retryable error or retries run out."""


@dataclass
class ReasoningOptions:
    tier: Literal["swift", "balanced", "frontier"] = "swift"
    effort: Literal["low", "medium", "high", "max"] = "medium"

    def as_payload(self) -> dict:
        return {"tier": self.tier, "effort": self.effort}


@dataclass
class EmbedStats:
    """Accumulated across every request made through one client."""

    requests: int = 0
    inputs: int = 0
    reasoning_cache_hits: int = 0
    reasoning_uncached: int = 0
    total_latency_ms: float = 0.0
    retries: int = 0

    def as_dict(self) -> dict:
        return {
            "requests": self.requests,
            "inputs": self.inputs,
            "reasoning_cache_hits": self.reasoning_cache_hits,
            "reasoning_uncached": self.reasoning_uncached,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "retries": self.retries,
        }


@dataclass
class PaprEmbeddingsClient:
    base_url: str = field(
        default_factory=lambda: os.getenv("PAPR_BASE_URL", "https://memory.papr.ai")
    )
    api_key: str = field(default_factory=lambda: os.getenv("PAPR_API_KEY", ""))
    timeout_s: float = 180.0
    max_attempts: int = 6
    stats: EmbedStats = field(default_factory=EmbedStats)

    def __post_init__(self) -> None:
        if not self.api_key:
            raise PaprAPIError(
                "PAPR_API_KEY is not set. Generate a key at https://dashboard.papr.ai "
                "(Settings -> API Keys) and export it before running."
            )
        self._http = httpx.Client(
            base_url=self.base_url.rstrip("/"),
            headers={"X-API-Key": self.api_key, "Content-Type": "application/json"},
            timeout=self.timeout_s,
        )

    # ------------------------------------------------------------------ API

    def list_models(self) -> list[dict]:
        response = self._request("GET", "/v1/embeddings/models")
        return response.json()["data"]

    def embed(
        self,
        texts: Sequence[str],
        *,
        model: str,
        input_type: InputType,
        schema_id: Optional[str] = None,
        reasoning: Optional[ReasoningOptions] = None,
    ) -> np.ndarray:
        """Embed ``texts`` (any length; batching is handled here) -> (n, dim)."""
        vectors: list[list[float]] = []
        for batch in _batches(texts):
            payload: dict = {
                "model": model,
                "input": list(batch),
                "input_type": input_type,
            }
            if schema_id is not None:
                payload["schema_id"] = schema_id
            if reasoning is not None:
                payload["reasoning"] = reasoning.as_payload()

            response = self._request("POST", "/v1/embeddings", json=payload)
            body = response.json()
            data = sorted(body["data"], key=lambda item: item["index"])
            vectors.extend(item["embedding"] for item in data)

            self.stats.requests += 1
            self.stats.inputs += len(batch)
            meta = body.get("meta", {})
            self.stats.total_latency_ms += float(meta.get("latency_ms") or 0.0)
            reasoning_meta = meta.get("reasoning") or {}
            self.stats.reasoning_cache_hits += int(reasoning_meta.get("cache_hits") or 0)
            self.stats.reasoning_uncached += int(reasoning_meta.get("uncached") or 0)

        return np.asarray(vectors, dtype=np.float32)

    # ------------------------------------------------------------- internals

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        last_error: Optional[str] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._http.request(method, path, **kwargs)
            except httpx.HTTPError as exc:
                last_error = f"transport error: {exc}"
            else:
                if response.status_code == 200:
                    return response
                # 4xx other than 429 will not improve on retry.
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    raise PaprAPIError(
                        f"{method} {path} -> {response.status_code}: {response.text[:500]}"
                    )
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"

            if attempt < self.max_attempts:
                self.stats.retries += 1
                # Exponential backoff with jitter; reasoning + cold GPU paths
                # legitimately return 503 while replicas warm.
                delay = min(60.0, (2.0**attempt) + random.uniform(0.0, 1.0))
                time.sleep(delay)

        raise PaprAPIError(
            f"{method} {path} failed after {self.max_attempts} attempts ({last_error})"
        )


def _batches(texts: Sequence[str]) -> Iterable[Sequence[str]]:
    """Split by both the 64-text and the combined-character server limits.

    A single text longer than the character budget is sent alone and left to
    the server's own validation (it truncates at the model's max context).
    """
    batch: list[str] = []
    chars = 0
    for text in texts:
        text_len = len(text)
        if batch and (len(batch) >= MAX_BATCH_SIZE or chars + text_len > CHAR_BUDGET):
            yield batch
            batch, chars = [], 0
        batch.append(text)
        chars += text_len
    if batch:
        yield batch

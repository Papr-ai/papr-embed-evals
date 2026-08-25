"""MTEB 2.x model wrapper that scores through the Papr embeddings API.

MTEB 2.10 calls ``encode`` with a ``DataLoader`` of batched inputs and a
``prompt_type`` telling us whether the batch is queries or corpus documents;
we map that straight onto the API's ``input_type``, which controls both the
server-side instruction (queries get the schema-resolved task instruction)
and — when reasoning is on — the teacher frame (queries are extracted
predictively, documents literally).

IMPORTANT: for query batches we send ``batch["query"]`` (the raw query), NOT
``batch["text"]`` — mteb's ``text`` field bakes in mteb's own instruction
template, and the Papr server applies the schema instruction itself. Sending
``text`` would double-instruct every query.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .client import PaprEmbeddingsClient, ReasoningOptions
from .tasks import TASKS

logger = logging.getLogger(__name__)


def _build_model_meta(model_name: str, embed_dim: int | None):
    """Minimal honest ModelMeta: the API is closed-weights, instruction-using."""
    from mteb.models.model_meta import ModelMeta

    return ModelMeta(
        loader=None,
        name=f"papr/{model_name}",
        revision="api",
        release_date="2026-08-20",
        languages=["eng-Latn"],
        n_parameters=None,
        memory_usage_mb=None,
        max_tokens=2048,
        embed_dim=embed_dim,
        license=None,
        open_weights=False,
        public_training_code=None,
        public_training_data=None,
        framework=["API"],
        similarity_fn_name="cosine",
        use_instructions=True,
        training_datasets=None,
    )


class PaprEmbedAPIModel:
    """MTEB 2.10 encoder: everything happens over the API."""

    def __init__(
        self,
        task_name: str,
        *,
        model_id: Optional[str] = None,
        reasoning: Optional[ReasoningOptions] = None,
        reasoning_queries_only: bool = False,
        client: Optional[PaprEmbeddingsClient] = None,
    ) -> None:
        spec = TASKS.get(task_name)
        if spec is None:
            raise ValueError(
                f"Unknown task {task_name!r}; add it to papr_embed_evals.tasks first "
                "(schema + instruction provenance is mandatory)."
            )
        self.spec = spec
        self.model_id = model_id or spec.model
        self.reasoning = reasoning
        # DEFAULT: reasoning hits BOTH queries and documents. The band cosine
        # compares band k of the query against band k of the doc, so a teacher
        # on one side meeting student bands on the other is a mismatched
        # comparison that measurably underprices the channel (see the V58
        # teacher-injection probe: both-sides at calibrated mass = +5.5 NDCG
        # mean vs student bands). Queries-only is a cost-bounding EXCEPTION
        # (teacher cost |queries| instead of |corpus|) and must be opted into
        # explicitly; results carry a -qonly suffix so the frames never mix.
        self.reasoning_queries_only = reasoning_queries_only and reasoning is not None
        if self.reasoning_queries_only:
            logger.warning(
                "[papr-api] QUERIES-ONLY reasoning: documents keep student "
                "bands. This underprices the teacher channel; use it for cost "
                "bounding, never as the headline teacher number."
            )
        self.client = client or PaprEmbeddingsClient()
        suffix = ""
        if reasoning:
            suffix = f"-reasoning-{reasoning.tier}-{reasoning.effort}"
            if self.reasoning_queries_only:
                suffix += "-qonly"
        self.model_name = f"{self.model_id}{suffix}"
        # MTEB reads this for result metadata (name/revision in the output).
        self.mteb_model_meta = _build_model_meta(
            self.model_name,
            embed_dim=6656 if "4b" in self.model_id else 3968,
        )

    # MTEB 2.10 encoder interface -------------------------------------------

    def encode(
        self,
        inputs: Any,
        *,
        task_name: str | None = None,
        task_metadata: Any = None,
        hf_split: str | None = None,
        hf_subset: str | None = None,
        prompt_type: Any = None,
        **_: Any,
    ) -> np.ndarray:
        input_type = "query" if _is_query(prompt_type) else "document"
        reasoning = self.reasoning
        if self.reasoning_queries_only and input_type != "query":
            reasoning = None

        def embed_one(texts: list[str]) -> np.ndarray:
            return self.client.embed(
                texts,
                model=self.model_id,
                input_type=input_type,
                # Pinned-schema models ignore this server-side.
                schema_id=self.spec.schema_id,
                reasoning=reasoning,
            )

        text_chunks = [t for t in _iter_texts(inputs, input_type) if t]
        if not text_chunks:
            return np.zeros((0, 0), dtype=np.float32)

        # PAPR_EMBED_CONCURRENCY > 1 keeps that many requests in flight. The
        # win is on the 4b path, whose base tower is a hosted multi-replica
        # endpoint: one serial stream leaves all but one replica idle. Results
        # come back in submission order (executor.map), so vectors still line
        # up 1:1 with the input rows. The model is batch-invariant, so request
        # composition cannot change any vector.
        concurrency = max(1, int(os.getenv("PAPR_EMBED_CONCURRENCY", "1")))
        chunks: list[np.ndarray] = []
        total = 0

        def note_progress(n: int) -> None:
            nonlocal total
            total += n
            if total % 6400 < n:
                logger.info(
                    "[papr-api] %s: %d %ss encoded (model=%s reasoning=%s conc=%d)",
                    task_name or self.spec.mteb_task,
                    total,
                    input_type,
                    self.model_id,
                    bool(reasoning),
                    concurrency,
                )

        if concurrency > 1 and len(text_chunks) > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for i, arr in enumerate(pool.map(embed_one, text_chunks)):
                    chunks.append(arr)
                    note_progress(len(text_chunks[i]))
        else:
            for texts in text_chunks:
                chunks.append(embed_one(texts))
                note_progress(len(texts))
        mat = np.concatenate(chunks, axis=0)

        # PAPR_DUMP_DIR: persist the returned vectors keyed by text sha1. The
        # export layout is public ([base | bands | evidence]), so a single
        # dumped run yields the entire band-mass surface offline by block
        # re-weighting -- no second paid pass against the API.
        dump_dir = os.getenv("PAPR_DUMP_DIR")
        if dump_dir:
            keys = [
                hashlib.sha1(t.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
                for texts in text_chunks
                for t in texts
            ]
            out = Path(dump_dir)
            out.mkdir(parents=True, exist_ok=True)
            path = out / (
                f"{self.model_name}_{task_name or self.spec.mteb_task}"
                f"_{input_type}_{int(time.time())}.npz"
            )
            np.savez_compressed(
                path, keys=np.array(keys), embs=mat.astype(np.float16)
            )
            logger.info("[papr-api] dumped %d vectors -> %s", len(keys), path)
        return mat

    # Embeddings are L2-normalized server-side, so cosine == dot product.

    def similarity(self, embeddings1: Any, embeddings2: Any) -> np.ndarray:
        a = np.asarray(embeddings1, dtype=np.float32)
        b = np.asarray(embeddings2, dtype=np.float32)
        if a.ndim == 1:
            a = a[None, :]
        if b.ndim == 1:
            b = b[None, :]
        return a @ b.T

    def similarity_pairwise(self, embeddings1: Any, embeddings2: Any) -> np.ndarray:
        a = np.asarray(embeddings1, dtype=np.float32)
        b = np.asarray(embeddings2, dtype=np.float32)
        return np.sum(a * b, axis=-1)


def _iter_texts(inputs: Any, input_type: str):
    """Yield lists of raw texts from either a DataLoader (mteb 2.10) or a
    plain sequence of strings (legacy callers / smoke tests)."""
    if isinstance(inputs, (list, tuple)) and (not inputs or isinstance(inputs[0], str)):
        yield list(inputs)
        return
    for batch in inputs:
        if isinstance(batch, dict):
            if input_type == "query" and batch.get("query") is not None:
                yield [str(text) for text in batch["query"]]
            else:
                yield [str(text) for text in batch["text"]]
        else:
            yield [str(text) for text in batch]


def _is_query(prompt_type: Any) -> bool:
    if prompt_type is None:
        # Retrieval tasks always pass prompt_type; None means a non-retrieval
        # caller, where the document path (no instruction) is the safe frame.
        return False
    value = getattr(prompt_type, "value", prompt_type)
    return str(value).lower() == "query"

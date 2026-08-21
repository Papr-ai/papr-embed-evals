"""MTEB 2.x model wrapper that scores through the Papr embeddings API.

MTEB calls ``encode`` with ``prompt_type`` telling us whether the sentences
are queries or corpus documents; we map that straight onto the API's
``input_type``, which controls both the server-side instruction (queries
get the schema-resolved task instruction) and — when reasoning is on — the
teacher frame (queries are extracted predictively, documents literally).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

from .client import PaprEmbeddingsClient, ReasoningOptions
from .tasks import TASKS

logger = logging.getLogger(__name__)


class PaprEmbedAPIModel:
    """Duck-typed MTEB encoder: everything happens over the API."""

    def __init__(
        self,
        task_name: str,
        *,
        model_id: Optional[str] = None,
        reasoning: Optional[ReasoningOptions] = None,
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
        self.client = client or PaprEmbeddingsClient()
        # MTEB inspects this attribute for result metadata.
        suffix = "-reasoning" if reasoning else ""
        self.mteb_model_meta = None
        self.model_name = f"{self.model_id}{suffix}"

    # MTEB 2.x encoder interface -------------------------------------------

    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str | None = None,
        prompt_type: Any = None,
        **_: Any,
    ) -> np.ndarray:
        input_type = "query" if _is_query(prompt_type) else "document"
        logger.info(
            "[papr-api] encoding %d %ss (model=%s schema=%s reasoning=%s)",
            len(sentences),
            input_type,
            self.model_id,
            self.spec.schema_id,
            bool(self.reasoning),
        )
        return self.client.embed(
            list(sentences),
            model=self.model_id,
            input_type=input_type,
            # Pinned-schema models ignore this server-side; harmless to send.
            schema_id=self.spec.schema_id,
            reasoning=self.reasoning,
        )


def _is_query(prompt_type: Any) -> bool:
    if prompt_type is None:
        # Retrieval tasks always pass prompt_type; None means a non-retrieval
        # caller, where the document path (no instruction) is the safe frame.
        return False
    value = getattr(prompt_type, "value", prompt_type)
    return str(value).lower() == "query"

#!/usr/bin/env python3
"""Embed the 809 SciFact TRAIN claims so band mass can be fitted off-test.

Why this exists. SciFact ships 1,109 claims split 809 train / 300 test over one
shared 5,183-abstract corpus, and the two query sets are disjoint (verified: 0
overlapping query ids). MTEB scores the test split. So the train split is a
legitimate, already-public tuning set for a serving hyperparameter like the
band mass p_m -- fitting there and reporting test once is exactly what the
split is for, and it is the same discipline AGENTS.md already mandates for
CoSQA ("always tune on COSQA_SPLIT=train, never test").

The corpus is shared, which is what makes this nearly free: the document
vectors dumped by the ordinary test run are the same 5,183 vectors the train
queries rank against. Only the 809 query vectors are missing, and only on the
plain path -- no teacher extraction, so no gpt-5.6-sol spend.

One caveat this script cannot remove, and the reason to compare the two optima
rather than just trusting the train one: papr-embed-v1 was fine-tuned on
SciFact train. The base block is therefore better on train claims than on
unseen ones, which biases a train-fitted p_m DOWNWARD (base looks stronger than
it will be at test time, so the fit under-buys the band channel). That bias has
a known sign, so the train-vs-test comparison measures it instead of assuming
it away.

Cost note for the teacher path. The band cache holds the 300 TEST claims and
the 5,183 documents, so re-running an eval is free -- but the 809 TRAIN claims
have never been extracted, and the cache keys on (content, schema, role, tier,
effort). So `TIER=frontier EFFORT=max` here bills 809 fresh gpt-5.6-sol xhigh
extractions per schema. It does NOT bill per model: the cache is
model-agnostic, so whichever tower runs first pays and the second reads.

    PAPR_API_KEY=... MODEL=papr-embed-v1-0.6b \
      python scripts/embed_scifact_train_queries.py

    # teacher-side fit (bills 809 extractions on the first tower per schema)
    PAPR_API_KEY=... MODEL=papr-embed-v1-0.6b TIER=frontier EFFORT=max \
      python scripts/embed_scifact_train_queries.py
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import pyarrow as pa  # noqa: E402

from papr_embed_evals.client import ReasoningOptions  # noqa: E402
from papr_embed_evals.mteb_model import PaprEmbedAPIModel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("embed-train")

HF = Path("/home/shawkatkabbara/.cache/huggingface/datasets/mteb___scifact")
HF_REV = "d56462d0e63a25450459c4f213e49ffdb866f7f9"


def read_arrow(path: Path):
    with pa.memory_map(str(path), "r") as stream:
        try:
            return pa.ipc.open_file(stream).read_all()
        except Exception:
            stream.seek(0)
            return pa.ipc.open_stream(stream).read_all()


def main() -> int:
    train_qids = {
        str(r["query-id"])
        for r in read_arrow(HF / f"default/0.0.0/{HF_REV}/scifact-train.arrow").to_pylist()
        if int(r.get("score", 0)) > 0
    }
    texts = [
        (r.get("text") or "").strip()
        for r in read_arrow(HF / f"queries/0.0.0/{HF_REV}/scifact-queries.arrow").to_pylist()
        if str(r["_id"]) in train_qids and (r.get("text") or "").strip()
    ]
    logger.info("resolved %d train claims", len(texts))
    if len(texts) != len(train_qids):
        raise SystemExit(
            f"{len(train_qids)} train qids but {len(texts)} texts; refusing a partial dump"
        )

    tier, effort = os.getenv("TIER"), os.getenv("EFFORT")
    reasoning = ReasoningOptions(tier=tier, effort=effort) if tier and effort else None
    model = PaprEmbedAPIModel(
        "SciFact", model_id=os.getenv("MODEL") or None, reasoning=reasoning
    )
    schema_id = os.getenv("SCHEMA_ID")
    if schema_id:
        if model.model_id.endswith("-scifact"):
            raise SystemExit(
                f"SCHEMA_ID={schema_id} needs a schema-aware MODEL; "
                f"{model.model_id} pins its schema server-side"
            )
        model.spec = dataclasses.replace(model.spec, schema_id=schema_id)
        model.model_name = f"{model.model_name}-s{schema_id.rsplit(':', 1)[-1].replace('.', 'p')}"

    batch = int(os.getenv("BATCH", "8"))
    model.model_name = f"{model.model_name}-b{batch}"
    chunks = [texts[i : i + batch] for i in range(0, len(texts), batch)]

    dump = os.getenv("PAPR_DUMP_DIR")
    if not dump:
        raise SystemExit("PAPR_DUMP_DIR is required; the vectors are the whole point")

    logger.info(
        "encoding as model_name=%s schema=%s batches=%d",
        model.model_name, model.spec.schema_id, len(chunks),
    )
    started = time.time()
    # task_name lands in the dump filename. "SciFactTrain" keeps these vectors
    # from ever being globbed as a test-split dump by the existing sweep tools.
    mat = model.encode(chunks, task_name="SciFactTrain", prompt_type="query")
    logger.info(
        "encoded %s in %.1f min; api=%s",
        mat.shape, (time.time() - started) / 60, model.client.stats.as_dict(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

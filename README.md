# papr-embed-evals

Retrieval evaluation (MTEB / CoIR / SciFact) for the
[papr-embed-v1](https://memory.papr.ai) embedding model lineup, scored
entirely through the public Papr embeddings API — no model weights needed.
Anyone with an API key can reproduce every number in `results/`.

## Models under evaluation

| model | base | what it is |
|---|---|---|
| `papr-embed-v1-0.6b-scifact` | Qwen3-0.6B | SciFact-specialized holographic embeddings |
| `papr-embed-v1-0.6b` | Qwen3-0.6B | general schema-aware embeddings |
| `papr-embed-v1-4b` | Qwen3-4B | larger base tower |
| `*-reasoning` | + live teacher | replaces the learned band block with request-time LLM-extracted schema bands |

Each model accepts `input_type` (`query` / `document`), a `schema_id`
selecting the holographic schema (pinned for scifact), and an optional
`reasoning: {tier, effort}` block. See `docs/METHODOLOGY.md` for what the
embedding vector actually is and why every schema/instruction choice here is
a measured decision.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Get an API key

1. Sign up at [dashboard.papr.ai](https://dashboard.papr.ai).
2. Settings → API Keys → Create key.
3. `export PAPR_API_KEY=<your key>`

Optional: `export PAPR_BASE_URL=...` to point at a non-production deployment.

## Run

```bash
# 1. pre-flight: key, models, reasoning path (seconds)
python scripts/smoke_api.py

# 2. SciFact, plain + reasoning A/B (the first real eval; ~90 batches/variant)
python scripts/run_scifact.py

# 3. CoIR, cheapest task first
ONLY="CodeTransOceanDL CodeTransOceanContest" python scripts/run_coir.py
python scripts/run_coir.py                 # full suite (mind the cost table)
REASONING=1 python scripts/run_coir.py     # reasoning variant
```

Results land in `results/<suite>/summary.json` plus the standard MTEB output
folders, ready for an MTEB leaderboard submission.

## Published results: official MTEB SciFact

NDCG@10 on the official test split (300 claims × 5,183 abstracts), `mteb` 2.10.0,
dataset revision `d56462d0e63a25450459c4f213e49ffdb866f7f9`. Every number is
scored through the public `memory.papr.ai/v1/embeddings` route — no local
checkpoint, no bespoke scoring path. Frozen-trunk baselines are the official MTEB
results-repo values for the stock Qwen3 models this lineup fine-tunes.

| Model | Schema | Plain | `frontier`/`max` | vs frozen trunk |
|---|---|---:|---:|---:|
| Qwen3-Embedding-0.6B *(frozen trunk)* | — | 69.718 | — | — |
| `papr-embed-v1-0.6b` | 2.0.0 | 71.476 | 71.516 | +1.80 |
| `papr-embed-v1-0.6b` | 3.1.0 | 70.884 | 71.245 | +1.53 |
| **`papr-embed-v1-0.6b-scifact`** | 2.0.0 | 72.990 | **73.147** | **+3.43** |
| Qwen3-Embedding-4B *(frozen trunk)* | — | 78.333 | — | — |
| **`papr-embed-v1-4b`** | 2.0.0 | **78.777** | 78.326 | **+0.44** |
| `papr-embed-v1-4b` | 3.1.0 | 78.620 | 78.593 | +0.29 |

`papr-embed-v1-4b` at 78.777 is above Qwen3-Embedding-8B (78.457) with half the
parameters. Schema `2.0.0` is the shipped default: it wins three of the four
general legs, and both headline numbers are 2.0.0.

### Reproduce exactly

```bash
export PAPR_API_KEY=<your key>

# 4B headline (78.777) — plain, schema 2.0.0
MODEL=papr-embed-v1-4b BATCH=32 ./scripts/queue_scifact_official_frontier.sh plain

# 0.6B headline (73.147) — SciFact-specialized, teacher on both sides
MODEL=papr-embed-v1-0.6b-scifact ./scripts/queue_scifact_official_frontier.sh both

# general 0.6B, schema 3.1.0 instead of the pinned default
MODEL=papr-embed-v1-0.6b SCHEMA_ID=biomedical:scifact:3.1.0 \
  ./scripts/queue_scifact_official_frontier.sh both
```

Each run writes standard MTEB JSON to
`results/scifact/<slug>/papr__<model>/api/SciFact.json`. Teacher extractions are
cached server-side by `(content, schema, role, tier, effort)` and the cache is
model-agnostic, so a `frontier/max` leg on a corpus another model already
extracted costs nothing — the 0.6b-scifact teacher run above reported
`reasoning_cache_hits: 5483, reasoning_uncached: 0`.

### Training-data disclosure

`papr-embed-v1` fine-tunes a **frozen** Qwen3 trunk on a 9-schema mix. Six of
those correspond to MTEB tasks and their **train** splits are in the mix:
`SciFact`, `NFCorpus`, `NQ`, `HotpotQA`, `FEVER`, `FiQA2018`. This is declared in
`training_datasets` on the emitted `ModelMeta`, so the leaderboard marks these as
in-domain. No test split is trained on, and no CoIR dataset is trained on — CoIR
is the sealed reporting axis. `papr-embed-v1-0.6b-scifact` is additionally a
task-specialized model (schema pinned to SciFact); read its number as an
in-domain result, not as evidence of general retrieval quality.

## Repo layout

```
papr_embed_evals/
  client.py       # API client: batching, retries, usage stats
  tasks.py        # per-task registry: schema, model, instruction provenance
  mteb_model.py   # MTEB 2.x encoder that scores through the API
schemas/          # vendored holographic schema JSONs (auditable band definitions)
scripts/          # smoke_api / run_scifact / run_coir
docs/METHODOLOGY.md  # provenance for every choice + fairness rules
results/          # eval outputs (committed for published runs)
```

## Status

- [x] API client + MTEB wrapper
- [x] Task registry with measured schema + instruction provenance
- [x] SciFact plain vs reasoning A/B — all 10 legs, production path
- [x] `ModelMeta` with declared training data, real param counts, correct dims
- [ ] Make this repo public (blocks the MTEB submission's reproducibility link)
- [ ] NFCorpus + CodeFeedbackMT
- [ ] CoIR suite through the API
- [ ] Switch transport to the `papr-memory` Python SDK once it exposes embeddings
- [ ] MTEB submission PR to `embeddings-benchmark/results`

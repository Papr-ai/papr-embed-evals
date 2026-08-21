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
- [ ] SciFact plain vs reasoning A/B (first run)
- [ ] CoIR suite through the API
- [ ] Switch transport to the `papr-memory` Python SDK once it exposes embeddings
- [ ] MTEB submission package

Private for now; will be open-sourced once the first submission-grade run is
published.

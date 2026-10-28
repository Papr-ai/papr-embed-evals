# papr-embed-evals

Reproduce the [papr-embed](https://huggingface.co/papr-ai) retrieval results
(MTEB / CoIR / SciFact). Two ways, matching the two model series:

| Series | How you run it | Script |
|---|---|---|
| **papr-embed-code-1** (4B, 0.6B), open weights, Apache-2.0 | Download from [Hugging Face](https://huggingface.co/papr-ai) and run locally on a GPU. Both models are registered in `mteb`. | `scripts/run_code1.py` |
| **papr-embed-schema-1** (4B, 0.6B), schema-native | Call the [Papr embeddings API](https://platform.papr.ai) with an API key. No weights needed. | `scripts/run_scifact.py`, `scripts/run_coir.py` |

## Quick start: code-1 (open weights)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-code1.txt
python scripts/run_code1.py --model papr-ai/papr-embed-code-1-0.6b --tasks coir   # CoIR mean 77.16
python scripts/run_code1.py --model papr-ai/papr-embed-code-1-4b   --tasks coir   # CoIR mean 80.12
```

The script loads the model through the entry that ships in `mteb`, so what it scores is what the
leaderboard scores. The entry's default context is 8,192 tokens. Six tasks have documents longer
than that (Apps, StackOverflowQA, CodeFeedbackMT, both CodeSearchNet variants, SWEbench), and the
published numbers run those at 32,768 with a small batch, as MTEB's maintainers asked:

```bash
python scripts/run_code1.py --model papr-ai/papr-embed-code-1-4b --max-len 32768 --batch 4 \
  --tasks AppsRetrieval,StackOverflowQA,CodeFeedbackMT,CodeSearchNetCCRetrieval,COIRCodeSearchNetRetrieval
```

On an 80 GB card in fp32, SWEbench at 32k needs `--batch 1`. Results are written in the
standard `mteb` layout. The weights, prompts and full protocol are on the model pages:
[4B](https://huggingface.co/papr-ai/papr-embed-code-1-4b) ·
[0.6B](https://huggingface.co/papr-ai/papr-embed-code-1-0.6b).

## Published results: code-1

nDCG@10, fp32, `mteb` 2.26.27, model revisions `2abaf9b8` (4B) and `43f82f3a` (0.6B). Every number
here has its result file under [`results/code1/`](results/code1/) in the standard `mteb` layout,
with a `run_settings.jsonl` line per run (batch size, devices, versions). The same files are
submitted to the MTEB results repo in
[embeddings-benchmark/results#813](https://github.com/embeddings-benchmark/results/pull/813).
Rankings below are against the MTEB results repo as of 26 Oct 2026 and will move.

**What was trained on.** code-1 was trained on the *train* splits of the CoIR datasets, CodeSearchNet,
CodeEditSearch, SciFact and several general retrieval sets (MS MARCO, NQ, HotpotQA, FEVER, ...).
The full list is declared in `training_datasets` on the `mteb` entry, so the leaderboard marks those
tasks as in-domain. No test split was trained on. CoIR and Code (v1) are therefore **in-domain**
benchmarks for this model; RTEB Code (except Apps), CodeRAG and SWEbench are **held out**.

### CoIR (10 tasks, in-domain)

| Model | Params | CoIR mean |
|---|---|---:|
| **papr-embed-code-1-4b** | 4B | **80.12** |
| F2LLM-v2-14B | 14B | 79.94 |
| Qwen3-Embedding-4B *(base)* | 4B | 78.87 |
| **papr-embed-code-1-0.6b** | 0.6B | **77.16** |
| F2LLM-v2-0.6B | 0.6B | 77.66 |
| Qwen3-Embedding-0.6B *(base)* | 0.6B | 74.67 |

The 4B's lead over F2LLM-v2-14B is 0.18, which is inside run-to-run noise on SyntheticText2SQL;
read it as level with the best 14B model, not a clear win. The 0.6B is 0.50 behind F2LLM-v2-0.6B.

| CoIR task | code-1-4b | code-1-0.6b | Batch, context |
|---|---:|---:|---|
| Apps | 92.56 | 79.31 | 4, 32k |
| CodeSearchNet | 89.84 | 86.18 | 4, 32k |
| CodeFeedback-MT | 95.64 | 94.21 | 4, 32k |
| CodeFeedback-ST | 90.10 | 88.52 | 16, 8k |
| CodeSearchNet-CC | 95.60 | 91.91 | 4, 32k |
| CodeTransOcean-Contest | 93.02 | 90.93 | 32, 8k |
| CodeTransOcean-DL | 35.14 | 34.47 | 32, 8k |
| CosQA | 40.29 | 40.42 | 32, 8k |
| StackOverflowQA | 96.72 | 93.57 | 4, 32k |
| SyntheticText2SQL | 72.27 | 72.04 | 32, 8k |
| **Mean** | **80.12** | **77.16** | |

Moving the long-document tasks from 8k to 32k context changed the means by 0.01 (4B) and 0.03 (0.6B);
the largest single move was CodeSearchNet-CC on the 0.6B, 92.23 → 91.91.

### Code (v1) (12 tasks, mostly in-domain)

CoIR plus CodeEditSearch and CodeSearchNet: **4B 81.13**, ahead of F2LLM-v2-14B and C2LLM-7B
(both 80.75); **0.6B 77.43**.

| Extra task | code-1-4b | code-1-0.6b | Qwen3-Emb-4B / 0.6B |
|---|---:|---:|---|
| CodeEditSearch (in-domain) | 80.07 | 66.62 | 76.21 / 64.25 |
| CodeSearchNet (near-domain) | 92.34 | 90.96 | 92.26 / 90.77 |

### RTEB Code (public tasks, held out except Apps)

| Task | code-1-4b | code-1-0.6b |
|---|---:|---:|
| Apps (in-domain) | 92.56 | 79.31 |
| DS1000 | 67.88 | 63.82 |
| FreshStack | 46.96 | 38.65 |
| HumanEval | 99.17 | 97.48 |
| MBPP | 93.57 | 90.80 |
| WikiSQL | 91.79 | 96.08 |
| **Mean (6)** | **81.99** | **77.69** |
| SWEbench (32k, not in the mean) | 7.89 | 5.26 |

This is **not** a leading result: the top RTEB Code models score about 4 points higher. Library
documentation (DS1000, FreshStack) is the weak spot. The two private RTEB Code tasks can only be
scored by MTEB's maintainers.

**SWEbench** (500 GitHub issues → the Python files the fix changed, out of 58k files) is hard for
every model. The 4B is 2nd of 13 entries, 0.52 behind bekko-embedding-v1-a25m (8.41) and just above
the BM25 baseline (7.76). Its Recall@100 (74.1) is the best of the entries we checked
(bekko 58.5, BM25 56.0): it finds the right files but rarely ranks them first.

### CodeRAG (3 tasks, held out)

| Task | code-1-4b | splade-code-8B | code-1-0.6b | splade-code-0.6B |
|---|---:|---:|---:|---:|
| ProgrammingSolutions | **80.00** | 79.46 | **77.05** | 76.15 |
| LibraryDocumentation | 66.39 | **69.33** | 61.92 | **62.87** |
| OnlineTutorials | **86.06** | 84.34 | **82.42** | 76.83 |
| **Mean** | 77.48 | **77.71** | **73.80** | 71.95 |

The 4B is 2nd of 11 on the mean and 1st on two of three tasks; the 0.6B leads its size class. Only
nine other models have CodeRAG scores, and none of the large general-purpose embedders are among
them, so read this as a thin comparison. CodeRAG's fourth task (StackoverflowPosts, 47M documents)
was not run; no model in the results repo has it.

### SciFact (in-domain for code-1)

| Model | SciFact | Qwen3-Embedding base | Gain |
|---|---:|---:|---:|
| papr-embed-code-1-4b | 86.79 | 78.33 | +8.46 |
| papr-embed-code-1-0.6b | 82.42 | 69.72 | +12.70 |

code-1 was trained on SciFact's train split, so this is an **in-domain** number and is **not**
comparable to the schema-1 SciFact table further down, where SciFact is closer to zero-shot.

## Quick start: schema-1 (through the API)

The rest of this README covers schema-1: `PAPR_API_KEY`, `scripts/smoke_api.py`, SciFact and CoIR runs.
Docs and API keys: [platform.papr.ai](https://platform.papr.ai).

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

## Experiment: schema-1 on CoIR through the API (partial)

Before code-1 existed, we scored schema-1 (then named `papr-embed-v1`) on three CoIR tasks through
the API, with a code schema per task (`code_search:*` in `papr_embed_evals/tasks.py`). These are
**not** code-1 results and **not** a full CoIR run (3 of 10 tasks), so there is no CoIR mean.
The files are in [`results/coir/`](results/coir/) under the model names used at the time.

| Task | schema-1-4b | schema-1-0.6b | 0.6b + reasoning (balanced / swift) |
|---|---:|---:|---|
| StackOverflowQA | 93.72 | 88.63 | 88.89 / 88.86 |
| SyntheticText2SQL | 77.59 | 76.61 (76.72 at batch 32) | 76.54 / 76.56 |
| CodeFeedbackMT | — | 87.90 | — |

For code retrieval, use code-1.

## Repo layout

```
papr_embed_evals/
  client.py       # API client: batching, retries, usage stats
  tasks.py        # per-task registry: schema, model, instruction provenance
  mteb_model.py   # MTEB 2.x encoder that scores through the API
schemas/          # vendored holographic schema JSONs (auditable band definitions)
scripts/          # smoke_api / run_scifact / run_coir
docs/METHODOLOGY.md  # provenance for every choice + fairness rules
results/
  code1/          # code-1 (open weights) result files, mteb layout
  scifact/        # schema-1 SciFact runs through the API
  coir/           # schema-1 partial CoIR experiment (3 tasks), not code-1
```

## Status

- [x] API client + MTEB wrapper
- [x] Task registry with measured schema + instruction provenance
- [x] SciFact plain vs reasoning A/B — all 10 legs, production path
- [x] `ModelMeta` with declared training data, real param counts, correct dims
- [x] Make this repo public (blocks the MTEB submission's reproducibility link)
- [ ] NFCorpus + CodeFeedbackMT
- [x] code-1: CoIR, Code (v1), RTEB Code, CodeRAG, SWEbench published (`results/code1/`)
- [ ] CoIR suite through the API for schema-1 (3 of 10 tasks done; superseded by code-1 for code)
- [ ] Switch transport to the `papr-memory` Python SDK once it exposes embeddings
- [x] MTEB submission PRs: results [#691](https://github.com/embeddings-benchmark/results/pull/691), ModelMeta [#5315](https://github.com/embeddings-benchmark/mteb/pull/5315)

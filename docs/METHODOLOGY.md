# Evaluation methodology and provenance

Every schema version, instruction string, and policy in `papr_embed_evals/tasks.py`
was chosen by a measurement, not a guess. This file records which measurement,
so a reviewer (or MTEB) can audit each choice without access to our internal
training repo.

## 1. What a papr-embed-v1 embedding is

The models export a single unit-norm vector

```
z(x) = [ sqrt(1 - p_m - p_e) * base(x) ; sqrt(p_m) * bands(x) ; sqrt(p_e) * evidence(x) ]
```

- `base(x)`: frozen Qwen3-Embedding trunk (0.6B or 4B), last-token pooled,
  official instruction recipe.
- `bands(x)`: a 14-band holographic block, one 128-d row per frequency band of
  the task's **schema**. The schema defines what each band means (e.g. for
  SciFact: claim stance, study type, organism, ...).
- `evidence(x)`: a 23-row evidence block (same construction, evidence-focused).
- `p_m, p_e`: per-schema mass fractions calibrated OFF-TEST on dev slices of
  the training data (never on the evaluation test sets).

Cosine similarity between two such vectors decomposes into exactly the
mass-weighted sum of the three block cosines, so retrieval scores are
reproducible from the API output alone.

## 2. Schema versions (the "which schema" decision)

| task | schema served | why |
|---|---|---|
| SciFact | `biomedical:scifact:2.0.0` | pinned by the scifact model |
| CosQA | `code_search:cosqa:2.0.0` | 2.0.0 redesign shipped after winning its A/B |
| all other CoIR | `<task>:1.0.0` | 1.1.0 revisions exist for six tasks but stay OFF until a measured win (policy: no version bump without an A/B) |

The full schema JSONs are vendored under `schemas/` so the band definitions
are auditable.

## 3. Query instructions (the "which prompt" decision)

The single largest variable found in our CoIR comparison was the query
instruction string. MTEB registers no prompt for any CoIR task, so baselines
fall back to the generic `"Retrieve text based on user query."`; our serving
stack resolves a bespoke per-task instruction from the schema. A parity study
(V55A, 2026-08-12) measured both prompts per task at matched context:

| effect of the generic prompt vs bespoke | tasks |
|---|---|
| large gain | AppsRetrieval **+2.82** |
| moderate gain | CodeFeedbackST +0.71, CodeTransOceanContest +0.60, StackOverflowQA +0.52, CosQA +0.48 |
| flat | CodeTransOceanDL −0.05, CodeFeedbackMT −0.11 |
| large loss | **SyntheticText2SQL −2.15**, CodeSearchNetCC −1.78 |

The usable rule extracted from this: **where the task is in-distribution for
the model's schemas, the bespoke instruction wins; where the task is foreign,
the phrasing the instruction-tuned trunk was trained on wins.** The
`prompt_winner` column in `tasks.py` records the measured winner per task.

The instruction template is the official Qwen one:

```
Instruct: {task_description}
Query:{query}
```

**It is applied server-side.** Clients send raw query text; the API resolves
the instruction from the schema. Never prepend an instruction client-side —
it would be applied twice.

### API gap (open)

Tasks where the *generic* prompt wins (Apps, Contest, CosQA, CodeFeedbackST,
StackOverflowQA) cannot currently be expressed through the public API, which
always serves the schema instruction. Known headroom ~+0.5 to +2.8 NDCG per
task. Fix under consideration: an optional `instruction` override field on
the embeddings route.

## 4. Reasoning-boosted embeddings (which "reasoning prompt")

`reasoning: {tier, effort}` replaces the student model's learned band block
with rows produced by a live LLM teacher at request time:

1. The teacher extracts the 14 schema-band values from the text.
2. **Queries are framed predictively** — the teacher does not describe the
   question; it predicts the metadata of the document that would answer it.
   Documents are extracted literally. This asymmetric framing is the "D"
   program that won the teacher-routing derivation (max-μ per schema over
   document-model × query-program; predictive-sol won or tied on 8 of 9
   training schemas).
3. Values are embedded and projected into the exact 14×128 basis the student
   was distilled toward (fixed seed, bit-identical to training).

The frame, provider, and projection are pinned server-side — callers choose
only `tier` (which teacher model class) and `effort`. Extractions are cached
per (content, schema, role), so query-vs-document never collide and re-runs
only pay for cache misses.

Default for evals: `tier=swift, effort=medium`. A/B against plain on SciFact
first (`scripts/run_scifact.py`), where the corpus is small and the band
machinery is deeply in-distribution.

## 5. Fairness rules (inherited from our MTEB eval recipe)

1. **Prompt parity**: any cross-model claim must state both models'
   instruction strings. A prompt difference is a prompt A/B wearing a model
   comparison's clothes.
2. **Context parity**: compare at matched max sequence length. (CodeFeedbackMT
   moved +0.96 NDCG from 2048→8192 purely because 8.6% of its queries
   truncate at 2048.)
3. **No test-set selection**: never choose a prompt, schema version, or mass
   value by comparing test-set NDCG. Choices are made on dev slices or by
   label-free rules, then frozen before the eval.
4. **Harness reproduction band**: deviations within ±0.6–1.0 NDCG of a
   published number are harness noise, not signal.

## 6. Cost model for API-driven evals

Embedding requests are capped at 64 texts / 100k characters. Corpus sizes:

| task | docs | ~document requests |
|---|---|---|
| SciFact | 5,183 | ~81 |
| CodeTransOceanDL | 816 | ~13 |
| CosQA | 20,604 | ~322 |
| CodeFeedbackMT | 66,383 | ~1,038 |
| CodeSearchNetCC | 280,652 | ~4,386 |
| COIRCodeSearchNet | ~1,005,474 | ~15,711 |

Reasoning multiplies document-side cost by teacher extraction (cache misses
only). Run SciFact and the small CoIR tasks first; the two CodeSearchNet
tasks need an explicit cost sign-off.

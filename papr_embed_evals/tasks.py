"""Per-task evaluation registry: schema, model, and instruction policy.

Every choice in this table is a MEASURED decision, not a default. Provenance
lives in docs/METHODOLOGY.md; the two source studies are the V55A instruction
parity evaluation (which found the query instruction was the single largest
variable in the whole CoIR comparison, worth +2.82 NDCG on AppsRetrieval and
-2.15 on SyntheticText2SQL depending on direction) and the schema-version
A/Bs (1.1.0 schema revisions exist for six CoIR tasks but stay OFF until a
measured win; CosQA's 2.0.0 redesign is the one shipped bump).

Instruction policy semantics
----------------------------
The Papr API applies the schema-resolved task instruction to queries ON THE
SERVER, so this harness always sends raw text. The ``instruction`` recorded
here is what the server resolves for that schema — kept in this table so the
eval is auditable without server access, and so MTEB reviewers can see the
exact string. ``prompt_winner`` records which prompt (bespoke vs the generic
"Retrieve text based on user query.") won the measured per-task A/B at 2048
context; tasks where the GENERIC prompt won cannot yet be expressed through
the public API (it always serves the schema instruction) — a known ~+0.5-2.8
NDCG headroom on five tasks, tracked in METHODOLOGY §"API gaps".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

GENERIC_INSTRUCTION = "Retrieve text based on user query."

# Models (papr-embed-v1 lineup). The scifact model pins its schema server-side.
SCIFACT_MODEL = "papr-embed-v1-0.6b-scifact"
GENERAL_MODEL = "papr-embed-v1-0.6b"
LARGE_MODEL = "papr-embed-v1-4b"


@dataclass(frozen=True)
class TaskSpec:
    mteb_task: str                      # exact MTEB 2.x task name
    schema_id: str                      # holographic schema served for this task
    model: str                          # default papr model for this task
    instruction: str                    # schema-resolved query instruction (server-side)
    prompt_winner: Literal["ours", "generic", "tie"]  # measured A/B winner @2048
    corpus_size: Optional[int] = None   # approximate, for cost planning
    notes: str = ""


TASKS: dict[str, TaskSpec] = {
    # ------------------------------------------------------------- SciFact
    "SciFact": TaskSpec(
        mteb_task="SciFact",
        schema_id="biomedical:scifact:2.0.0",
        model=SCIFACT_MODEL,
        instruction="Given a scientific claim, retrieve documents that support or refute the claim.",
        prompt_winner="ours",
        corpus_size=5_183,
        notes="Pinned schema; the model ignores caller schema_id. First target for reasoning A/B.",
    ),
    # ---------------------------------------------------------------- CoIR
    "AppsRetrieval": TaskSpec(
        mteb_task="AppsRetrieval",
        schema_id="code_search:apps:1.0.0",
        model=GENERAL_MODEL,
        instruction="Given a programming problem description, retrieve the code that solves the problem.",
        prompt_winner="generic",
        corpus_size=9_000,
        notes="Generic prompt measured +2.82 over bespoke; not expressible via API yet.",
    ),
    "CodeTransOceanContest": TaskSpec(
        mteb_task="CodeTransOceanContest",
        schema_id="code_search:codetrans_contest:1.0.0",
        model=GENERAL_MODEL,
        instruction="Given a program, retrieve the program in another language that implements the same functionality.",
        prompt_winner="generic",
        corpus_size=1_000,
    ),
    "CodeTransOceanDL": TaskSpec(
        mteb_task="CodeTransOceanDL",
        schema_id="code_search:codetrans_dl:1.0.0",
        model=GENERAL_MODEL,
        instruction=(
            "Given a deep learning code snippet written in one framework, retrieve the same "
            "algorithm implemented in a different deep learning framework; a near-identical "
            "snippet in the query's own framework is not the answer."
        ),
        prompt_winner="tie",
        corpus_size=816,
        notes="Anti-twin emphasis instruction (schema-lock phase 3).",
    ),
    "CosQA": TaskSpec(
        mteb_task="CosQA",
        schema_id="code_search:cosqa:2.0.0",
        model=GENERAL_MODEL,
        instruction="Given a natural language question, retrieve code snippets that best answer the question.",
        prompt_winner="generic",
        corpus_size=20_604,
        notes="The one task on the 2.0.0 schema redesign.",
    ),
    "CodeFeedbackST": TaskSpec(
        mteb_task="CodeFeedbackST",
        schema_id="code_search:codefeedback_st:1.0.0",
        model=GENERAL_MODEL,
        instruction="Given a coding question, retrieve the response that best answers it.",
        prompt_winner="generic",
        corpus_size=156_526,
    ),
    "StackOverflowQA": TaskSpec(
        mteb_task="StackOverflowQA",
        schema_id="code_search:stackoverflow_qa:1.0.0",
        model=GENERAL_MODEL,
        instruction="Given a technical question, retrieve the answer that resolves it.",
        prompt_winner="generic",
        corpus_size=19_931,
    ),
    "SyntheticText2SQL": TaskSpec(
        mteb_task="SyntheticText2SQL",
        schema_id="code_search:synthetic_text2sql:1.0.0",
        model=GENERAL_MODEL,
        instruction="Given a natural language question, retrieve the SQL query that answers the question.",
        prompt_winner="ours",
        corpus_size=105_851,
        notes="Bespoke prompt measured +2.15 over generic — deeply in-distribution schema.",
    ),
    "CodeFeedbackMT": TaskSpec(
        mteb_task="CodeFeedbackMT",
        schema_id="code_search:codefeedback_mt:1.0.0",
        model=GENERAL_MODEL,
        instruction="Given a multi-turn coding conversation, retrieve the response that best continues it.",
        prompt_winner="ours",
        corpus_size=66_383,
        notes="8.6% of queries exceed 2048 tokens; long-context matters here.",
    ),
    "COIRCodeSearchNetRetrieval": TaskSpec(
        mteb_task="COIRCodeSearchNetRetrieval",
        schema_id="code_search:codesearchnet:1.0.0",
        model=GENERAL_MODEL,
        instruction="Given a code snippet, retrieve the natural language summary that describes its functionality.",
        prompt_winner="ours",
        corpus_size=1_005_474,
        notes="~1M docs: run through the API only with a cost plan.",
    ),
    "CodeSearchNetCCRetrieval": TaskSpec(
        mteb_task="CodeSearchNetCCRetrieval",
        schema_id="code_search:codesearchnet_ccr:1.0.0",
        model=GENERAL_MODEL,
        instruction="Given a code snippet, retrieve the most relevant code snippet.",
        prompt_winner="ours",
        corpus_size=280_652,
        notes="Generic prompt measured -1.78 here — bespoke required.",
    ),
}

COIR_TASKS: list[str] = [name for name in TASKS if name != "SciFact"]

# Cheapest-first ordering for CoIR, so throughput and correctness are proven
# on small corpora before the six-figure-document tasks are committed to.
COIR_ORDER: list[str] = [
    "CodeTransOceanDL",
    "CodeTransOceanContest",
    "AppsRetrieval",
    "StackOverflowQA",
    "CosQA",
    "CodeFeedbackMT",
    "SyntheticText2SQL",
    "CodeFeedbackST",
    "CodeSearchNetCCRetrieval",
    "COIRCodeSearchNetRetrieval",
]

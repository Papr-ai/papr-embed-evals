#!/usr/bin/env python3
"""Score the open-weights papr-embed-code-1 models on CoIR (and SciFact) with stock mteb.

Unlike the schema-1 scripts in this repo (which call the Papr API), this one downloads the
weights from Hugging Face and runs them locally through the model entry that ships in mteb:

  pip install -r requirements-code1.txt
  python scripts/run_code1.py --model papr-ai/papr-embed-code-1-0.6b --tasks coir
  python scripts/run_code1.py --model papr-ai/papr-embed-code-1-4b   --tasks coir,SciFact --batch 16

Models: papr-ai/papr-embed-code-1-4b and papr-ai/papr-embed-code-1-0.6b
        (https://huggingface.co/papr-ai). A GPU is needed; fp32 reference precision.
Results are written in the standard mteb layout under --out, the same format as the
leaderboard's results repo. Published numbers: 4B CoIR 80.12, 0.6B CoIR 77.16 (see README).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mteb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default="papr-ai/papr-embed-code-1-4b",
                    help="papr-ai/papr-embed-code-1-4b or papr-ai/papr-embed-code-1-0.6b")
    ap.add_argument("--tasks", default="coir",
                    help="'coir' (the 10-task benchmark) or a comma-separated list of mteb task names")
    ap.add_argument("--out", default="results/code1", help="output folder (mteb cache layout)")
    ap.add_argument("--batch", type=int, default=32,
                    help="batch size. The long-document tasks need 4 at 32k context (1 for SWEbench) "
                         "on an 80 GB card in fp32")
    ap.add_argument("--max-len", type=int, default=None,
                    help="max sequence length. Default: the mteb entry's 8192. The published numbers "
                         "use 32768 for the tasks with documents over 8k tokens (Apps, StackOverflowQA, "
                         "CodeFeedbackMT, both CodeSearchNet variants, SWEbench)")
    a = ap.parse_args()

    meta = mteb.get_model_meta(a.model)  # entry merged into mteb; carries the loader and prompts
    print(f"[model] {meta.name} revision={meta.revision[:8]} mteb={mteb.__version__}", flush=True)
    model = mteb.get_model(a.model)
    if a.max_len:
        st = getattr(model, "model", model)  # the SentenceTransformer inside mteb's wrapper
        st.max_seq_length = a.max_len
        print(f"[model] max_seq_length={st.max_seq_length}", flush=True)

    names = [t.strip() for t in a.tasks.split(",") if t.strip()]
    tasks = []
    for n in names:  # 'coir' expands to the 10-task benchmark; anything else is an mteb task name
        tasks += list(mteb.get_benchmark("CoIR").tasks) if n.lower() == "coir" else mteb.get_tasks(tasks=[n])

    cache = mteb.ResultCache(cache_path=a.out)
    results = mteb.evaluate(model, tasks, cache=cache, encode_kwargs={"batch_size": a.batch})

    scores = {}
    for r in results.task_results:
        s = r.get_score()  # main score (nDCG@10), averaged over the task's subsets
        scores[r.task_name] = round(s * 100, 2)
        print(f"[task] {r.task_name}: nDCG@10 {s * 100:.2f}", flush=True)
    coir = [t.metadata.name for t in mteb.get_benchmark("CoIR").tasks]
    if all(t in scores for t in coir):
        scores["CoIR mean"] = round(sum(scores[t] for t in coir) / len(coir), 2)
        print(f"[CoIR mean] {scores['CoIR mean']:.2f}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(scores, open(Path(a.out) / f"{a.model.split('/')[-1]}_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()

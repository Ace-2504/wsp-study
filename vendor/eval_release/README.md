# Small-LM Legal / Financial QA — Evaluation Release

Anonymized reproducibility bundle for the paper's evaluation: the benchmark, the exact judge
prompt/rubric, and standalone scoring code. Everything needed to reproduce the reported numbers or
score a new model on the same benchmark.

> Double-blind note: this release is intentionally anonymous. Author identity, model repositories,
> and the live demo are omitted and will be added only in the camera-ready version.

## Contents

| File | What it is |
|---|---|
| `eval_set.jsonl` | The benchmark — 2,000 items (see conditions below) |
| `judge_prompt.md` | The LLM-judge rubric, output schema, and exact prompt construction |
| `score.py` | Standalone scorer: offline metrics + the optional rubric-10 LLM judge |
| `DATASET_CARD.md` | Dataset card (sources, sizes, intended use, limitations) |

## The benchmark

Grounded question answering over US legal / financial / general-web passages. Each item has a
source passage (context), a question, and a short gold answer. The set spans **four conditions**,
500 items each (2,000 total):

- **clean** — the correct supporting passage is provided (the main scored condition).
- **realistic** — a realistic retrieval mix (correct passage among plausible neighbours).
- **retrieval_failure** — the correct passage is absent (tests abstention / not-in-context).
- **closed_book** — no passage (tests parametric knowledge).

Each `eval_set.jsonl` line:

```json
{"id": "...", "source": "case-law|sec|fineweb-edu", "condition": "clean|realistic|retrieval_failure|closed_book",
 "system": "...", "context": "...", "question": "...", "gold_answer": "..."}
```

## Scoring a model

1. Produce a predictions file — one line per item, `{"id": "<eval id>", "answer": "<model output>"}`,
   using the same decoding as the paper (greedy, 160 new tokens) if you want comparable numbers.
2. Score:

```bash
# offline metrics only (token-F1 + fabrication proxy) — no API
python score.py --predictions preds.jsonl --condition clean

# add the rubric-10 LLM judge (OpenAI-compatible endpoint; --base-url for others)
python score.py --predictions preds.jsonl --condition clean \
    --judge --model <judge-model> --api-key <KEY>
```

Metrics: **rubric total /10** (correctness + completeness + groundedness + clarity), **grounded %**,
**token-F1**, and a reference-free **fabrication %**. The judge scores each answer independently at
temperature 0; see `judge_prompt.md` for the exact prompt.

## Reproducibility notes

- The judge is model-agnostic — any capable instruct model can fill the judge role via `score.py`;
  the paper reports a primary judge and a second independent judge for a cross-judge robustness check.
- `pip install openai` is only needed for the `--judge` path; the offline metrics use the standard
  library only.

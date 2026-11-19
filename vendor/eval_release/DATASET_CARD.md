# Dataset Card — Small-LM Legal / Financial QA Benchmark

## Summary

A grounded question-answering benchmark for small language models over US legal, financial, and
general-web text. Each item pairs a source passage with a question and a short gold answer, and is
presented under four retrieval conditions to probe both answering and abstention behaviour.

## Composition

- **Size:** 2,000 items — 500 in each of four conditions (`clean`, `realistic`,
  `retrieval_failure`, `closed_book`).
- **Sources** (of the underlying passages): US case law, SEC / financial filings, and educational
  web text. In the primary `clean` condition the source mix is roughly SEC 39% / case-law 31% /
  web 30%.
- **Fields:** `id`, `source`, `condition`, `system` (grading instruction), `context` (passage,
  empty in `closed_book`), `question`, `gold_answer`.

## Construction

Questions and gold answers were generated from the source passages and filtered for answerability.
The evaluation passages were **decontaminated** against the model's pretraining corpus (chunk-level
deduplication) so test items are not memorised training text. Gold answers are short (median ~12
tokens); questions are short (median ~18 tokens); passages are capped to a fixed window.

## Intended use

- Benchmarking small (≈0.1B–3B) language models on grounded QA and abstention.
- Studying how fine-tuning stages (supervised, retrieval-augmented, preference-based) change
  answer quality and fabrication.

Not intended for legal or financial advice, or for training safety-critical systems.

## Metrics

- **Rubric total (0–10):** correctness (0–5) + completeness (0–2) + groundedness (0–2) + clarity
  (0–1), scored by an LLM judge (see `judge_prompt.md`).
- **Grounded %:** share of answers with full groundedness (nothing invented).
- **Token-F1:** overlap with the gold answer (reference-based, no judge).
- **Fabrication %:** reference-free proxy — an answer states a number not supported by its context.

## Limitations

- LLM-judge scores are validated against human grading in the paper, but any automated judge has
  residual bias; report figures alongside the reference-free token-F1 and fabrication metrics.
- The retrieval-condition splits are constructed, not sampled from a production retrieval system.
- English, US-jurisdiction legal/financial text only.

## Licence

Passages derive from public datasets under their respective licences; consult the source datasets
before redistribution. The questions, gold answers, judge prompt, and scoring code in this release
are provided for research reproducibility.

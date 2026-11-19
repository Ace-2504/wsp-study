# Gate 1 Report — the instrument, proved before any edit

**Question:** can this workspace reproduce the published 125M QA results deterministically, and
what do the two originals (θ_e4, θ_qa) measure on every axis the WSP study will use, before a
single weight is touched?
**Answer:** **yes, the reproduction passes.** Building and running the instrument also produced
six findings that change the experimental design (§7).

No weight was edited. Every number below is computed by `python -m wsp.gate1_report` from saved
artifacts in `wsp_runs/gate1/` (tables: `wsp_runs/gate1/GATE1_TABLES.md`, raw:
`gate1_numbers.json`). Hardware: one RTX 3060 12 GB; torch 2.4.1+cu121, transformers 4.46.3,
the same library versions as the published runs.

---

## 1. Originals: pinned, locked, verified

| | HF repo | Revision | dtype | Lock |
|---|---|---|---|---|
| θ_e4 | `Ace-2504/slm-125m-e4` | `bf8ca36e` | fp32 | SHA-256 per file in `checkpoints/_original/LOCK.json`, files read-only, verified at the start of every run |
| θ_qa | `Ace-2504/slm-125m-sft` | `c3579e31` | bf16 | same |

Tokenizers are byte-identical. Configs differ only in `torch_dtype`.

## 2. Reproduction of θ_qa's published clean numbers — **PASS**

Instrument: `data/eval/eval_messages.jsonl`, SHA-256 `16e12b23…`, which is the exact file in
the published manifests. Greedy decoding, 160 new tokens, no BOS, one item at a time.

| metric (clean, n=500) | ours | published | paired Δ (ours − published) |
|---|---|---|---|
| token-F1 | 0.2286 [0.2058, 0.2515] | 0.2286 [0.2058, 0.2520] | +0.0000 [−0.0074, +0.0068] |
| exact match | 0.034 [0.018, 0.050] | 0.038 [0.022, 0.056] | −0.004 [−0.010, 0.000], n.s. |
| fabrication | 0.062 [0.042, 0.084] | 0.062 [0.042, 0.084] | +0.000 |
| false abstention | 0.000 | 0.000 | +0.000 |

- **446 of 500 generations (89.2%) are byte-identical** to the published ones, which were produced
  on a Modal L4. The other 54 diverge at a near-tied greedy token, so the two runs differ through
  bf16 rounding across GPUs. No metric moves beyond its CI.
- The metric code is a verbatim copy of the vendor's, and re-scoring all 3,000 stored reference
  items gives **0 score mismatches**.

**Consequence for the study:** a 10.8% cross-GPU flip rate is large next to the effects editing
may produce. **Every arm and its baseline must be scored on the same hardware.** Comparing a local
edit against a published Modal number would mix in this noise.

## 3. θ_qa on all four conditions (never measured before)

| condition | token-F1 | exact match | fabrication ↓ | abstain rate |
|---|---|---|---|---|
| clean (answer in context) | 0.229 | 0.034 | 0.062 | 0.000 |
| realistic (answer among distractors) | 0.198 | 0.022 | 0.044 | 0.000 |
| retrieval_failure (answer absent, should abstain) | 0.081 | 0.000 | 0.056 | **0.000** |
| closed_book (no context) | 0.154 | 0.000 | **0.188** | 0.000 |

**θ_qa never abstains.** On all 500 questions whose answer is not in the documents, it produces
an answer: its abstain rate is **0/500**, although the system prompt asks for "not stated in the
context". *Correction (Gate 2 data check):* θ_qa was **never trained to abstain**. `qa_sft.jsonl`
contains 0 abstention targets out of 15,000, and θ_qa never saw this eval's system prompt (the
RAFT "grounded assistant" prompt used by clean, realistic and retrieval_failure; closed_book uses
θ_qa's own training prompt). So abstention is a **missing behaviour to install**, not a weak one
to sharpen. Without context (closed_book) it invents a
number in **18.8%** of answers. These are the measurable "sharpen" targets with the most headroom.
Abstention in particular starts from a 0% floor, so any real movement is unambiguous.

*(θ_e4 3-shot on the same four conditions: **pending**. The run was still in progress when this
draft was written; see §8.)*

## 4. Does train/eval overlap inflate the published score?

`splits/eval_overlap.json`: 450 of the 500 clean eval items share **no** 13-gram with θ_qa's
training data, and 50 share some (15 share ≥10% of their passage).

| metric | overlapping (50) − clean (450) | significant |
|---|---|---|
| token-F1 | **+0.118** [+0.018, +0.216] | yes |
| exact match | **+0.096** [+0.013, +0.191] | yes |
| fabrication | +0.020 [−0.049, +0.107] | no |
| judged correctness (published judge) | +0.082 [−0.036, +0.208] | no |

**Verdict:** the contamination is real at the item level, since θ_qa does measurably better on
passages that resemble its training data. But at 50/500 items it barely moves the headline:
judged correctness is **0.261 on all 500 versus 0.253 [0.219, 0.286] on the clean 450**, well inside
the CI. **From now on every QA number is reported twice, on all 500 (for comparability with the
published study) and on the clean 450 (the uncontaminated score).** The 505 training rows behind
the overlap are quarantined from `edit-dev` (`splits/qa_sft_quarantine.json`), so no editing
decision is ever made on text that resembles the final test.

## 5. Retention: what fine-tuning already cost (the forgetting scale)

Per-window held-out next-token loss, same windows for both models (paired), 35,293 windows in
total. The two sets are **strictly disjoint**: v1's SEC bins turned out to be byte-identical to
e4's and 760 fineweb windows were shared, so those duplicates were removed (§7).

| set / domain | windows | θ_e4 | θ_qa | Δ (qa − e4) | Δ % | windows worse |
|---|---|---|---|---|---|---|
| e4-native / SEC | 8,485 | 1.6942 | 1.7479 | +0.0537 [+0.0532, +0.0542] | **+3.17%** | 99.9% |
| e4-native / case-law | 8,354 | 2.2451 | 2.3076 | +0.0625 [+0.0621, +0.0629] | +2.78% | 100.0% |
| e4-native / fineweb-edu | 7,581 | 3.1386 | 3.2232 | +0.0846 [+0.0841, +0.0850] | +2.69% | 100.0% |
| v1-heldout / case-law | 7,053 | 2.2189 | 2.2828 | +0.0639 [+0.0635, +0.0643] | +2.88% | 100.0% |
| v1-heldout / fineweb-edu | 3,820 | 3.0918 | 3.1768 | +0.0850 [+0.0844, +0.0857] | +2.75% | 100.0% |

Sanity check: θ_e4's losses agree with its model card (SEC 1.6977, case-law 2.2324, web 3.0987
on a related probe protocol) to within the expected protocol difference.

**Full QA fine-tuning raised held-out loss by about 3% uniformly**, and on essentially every single
window. This had never been measured for this lineage. It sets the scale for judging forgetting: an
edit that costs, say, +0.5% has done one sixth of the damage of the fine-tuning that built the skill.

## 6. Copy machinery and the task vector

**Phase-0 probes** (pure continuation, identical passages and codes, so the trials are paired):

| probe | θ_e4 | θ_qa | paired Δ |
|---|---|---|---|
| induction, random tokens (2nd-copy acc) | 0.839 | 0.820 | — |
| induction, natural text | 0.853 | 0.858 | — |
| **needle, verbatim cue** | 0.910 [0.870, 0.945] | **0.960** [0.930, 0.985] | **+0.050 [+0.020, +0.085] ✓** |
| needle, verbatim, control (no needle) | 0.000 | 0.000 | — |
| needle, raw "Question:/Answer:" cue | 0.040 | 0.050 | +0.010 [−0.025, +0.045] n.s. |
| needle, question, control | 0.000 | 0.000 | — |

QA fine-tuning **strengthened** verbatim copy-from-context (+5 pp, significant). That fits the
hypothesis from v1-extended's Phase 0: grounded SFT builds on the copy machinery. But the raw
`Question:/Answer:` cue is still at 4–5% after SFT, so **the QA skill θ_qa learned is bound to its
`<|user|>/<|assistant|>` template.** It is a prompt-conditioned behaviour, not a general
question→span mapping. For WSP this means probing has to happen inside the chat template.

**Task vector τ = θ_qa − θ_e4** (`wsp_runs/gate1/taskvec_stats.json`):
- **bf16 rounding share of τ: exactly 0.** e4's fp32 weights already lie on the bf16 grid, since
  e4 was trained in bf16 and saved upcast. So τ is entirely learned change.
- ‖τ‖ / ‖θ_e4‖ = **1.67%** over 125.8M parameters in 110 tensors, which means **110 coefficients**
  for per-tensor α search.
- The largest relative changes are in **layer 0–1 attention** (L0 o_proj 5.2%, v_proj 4.2%,
  k_proj 3.9%, q_proj 2.9%; L1 o_proj 2.7%). This is descriptive only, and Gate 1 does not interpret
  it. One plausible reading, to be tested, is that early attention is where the new chat-template
  role tokens get wired in.

## 7. Findings that change the design

1. **Abstention is the cleanest WSP target.** θ_qa scores 0/500 on retrieval_failure abstention,
   starting from a hard floor with full headroom, and errors there are fabrications by definition.
   I propose it as the **primary Part-I target**, with clean-condition correctness as the
   guardrail (§9). *Caveat found at Gate 2:* θ_qa was never trained to abstain (§3), so this is an
   **install** task, the regime where the literature expects weight editing to be weakest, and the
   one where LoRA should be strongest. The reference `slm-125m-raft` (e4 + full SFT on RAFT data)
   over-abstains on 84% of answerable items, so false abstention on clean items is the guardrail
   that matters most.
2. **The published "125M base" row is v1, not e4.** `base-125m` in the reference was
   `Ace-2504/slm-125m-base`. e4's QA floor is measured here for the first time (§8).
3. **Comparisons must be same-hardware.** The cross-GPU greedy flip rate is 10.8%.
4. **The "independent" retention sets weren't independent.** SEC files are identical and 760
   fineweb windows were shared. They are now strictly disjoint.
5. **Contamination is real but small.** QA scores are reported on all 500 and on the clean 450;
   505 training rows are quarantined.
6. **θ_qa's QA skill is template-bound.** WSP introspection (activations, gradients, attribution)
   must use the chat-template prompt format, not raw continuation.

Instrument caveat (minor): in the `realistic` condition prompts reach 884 tokens, so an answer longer
than ~140 tokens would run past 1,024 positions. θ_qa's answers are short, so this barely matters
here, but the WSP harness will cap new tokens at `1024 − prompt length`, and that deviation is
logged.

## 8. θ_e4 on the QA instrument (3-shot)

Same instrument and protocol as the published base row (first 3 `qa_sft` pairs as exemplars,
dropped one at a time to fit the context), 500 items per condition:

| condition | θ_e4 token-F1 | θ_e4 fabrication | θ_qa token-F1 | paired Δ token-F1 (qa − e4) |
|---|---|---|---|---|
| clean | 0.018 [0.016, 0.020] | 0.014 | 0.229 | **+0.211 [+0.188, +0.234]** |
| realistic | 0.016 | 0.102 | 0.198 | +0.182 [+0.163, +0.203] |
| retrieval_failure | 0.015 | 0.078 | 0.081 | +0.067 [+0.059, +0.074] |
| closed_book | 0.020 | 0.098 | 0.154 | +0.134 [+0.121, +0.147] |

Exact match and abstain rate are 0.000 for θ_e4 in every condition. **θ_e4's QA floor (clean token-F1
0.018) matches the published v1-base floor (0.019).** Four epochs of continued pretraining added no
measurable few-shot QA, so essentially all of θ_qa's QA ability comes from the fine-tune, and
**θ_e4 does not abstain either.** θ_e4's low clean fabrication (0.014) is not grounding: it
reflects how little it answers at all. Full tables: `wsp_runs/gate1/GATE1_TABLES.md`.

## 9. Proposed regression thresholds (pre-registration, for approval)

Each threshold comes from a measured scale, not a guess.

| Guardrail | Threshold for an edit of θ_qa | Derived from |
|---|---|---|
| Held-out loss, any domain | ≤ **+0.5%** vs θ_qa | 1/6 of the ~3% that full SFT cost (§5); paired CIs here are ±0.03%, so +0.5% is far above noise |
| SEC held-out loss | ≤ **+0.5%** (same bound; no looser for the family's one external win) | §5 |
| Clean token-F1 (paired, all 500) | lower CI bound of Δ ≥ **−0.02** | half-width of the clean CI at n=500 is ±0.023 |
| Clean judged correctness (milestones only) | Δ CI must not exclude zero downward | published judge, same rubric |
| Verbatim copy (needle) | ≥ **0.93** (θ_qa's lower CI bound) | §6 |
| Fabrication (clean) | must not rise beyond its CI (0.062 [0.042, 0.084]) | §3 |

An arm that breaches any guardrail fails, regardless of its gain on the target, and is rolled back.

## 10. Gate 1 verdict

**PASS** on reproduction, determinism and the measurement baseline. The instrument reproduces the
published θ_qa numbers on its own hardware, the originals are locked, and every axis of the
proposed study now has a measured zero point and noise floor.

---

## Addendum (Gate 2 data check, 2026-11-10): RAFT shares the eval's distractor documents

Checked before building the abstention pools from `raft.jsonl`:

| What | Found in RAFT prompts (≥90% same 13-grams) |
|---|---|
| Eval clean golden passages (answer-bearing) | **1 / 500** |
| Eval retrieval_failure documents (all distractors) | **1,000 / 1,000** (every item) |
| Eval realistic documents | 1,001 / 1,500 (every item has at least one) |
| Identical question strings, eval vs RAFT | 0 |

The eval's quarantine was applied at `chunk_id` level to *training pairs*, not to the *distractor
pool* (see `vizuara-fine-tuning/bugs/05`). Answers did not leak. But **every document in the
abstention condition was seen in RAFT training**. Consequences:

1. **The published `slm-125m-raft` abstention score (0.658 on retrieval_failure) is contaminated**
   at the document level. It may partly reflect "I have seen these documents in abstention
   prompts" rather than general abstention.
2. **The abstention pools for this study must exclude every RAFT row that contains an eval
   document.** Otherwise an edit or LoRA could learn document-specific "abstain" associations and
   inflate the primary metric. The strict rule (a row is quarantined if any of its documents shares
   ≥50% of its 13-grams with any eval document) keeps only **~1,489 of 10,000 rows**, giving about
   870 for edit-dev and four iter-val folds of about 155 each. `src/wsp/splits.py` currently
   implements the stricter "any shared 13-gram" rule (1,269 rows kept) and must be switched to the
   ≥50% per-document rule before use.
3. This strengthens the case for **generating a fresh `iter-val` set** (≈$1 Gemini) from corpus
   chunks that appear nowhere in the eval set.

## 🛑 GATE 1 ENDS HERE

**Green flag received 2026-11-10 (Harman):** the regression thresholds in §9 and abstention as the
primary Part-I target (§7.1) are **approved and locked as pre-registered**, before any edit. Still
open: the editor LLM (needed at Gate 3); `iter-val` generation may be unnecessary for the abstention
target, since Gate 2 checks whether held-out RAFT rows can serve.

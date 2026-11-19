# Deliverable 1 — Model Evaluation Audit

> **Revised — see [`05_LINEAGE_REVISION.md`](05_LINEAGE_REVISION.md).** This audit covers the
> pretraining lineage up to `slm-125m-extended`, and its findings still hold for that lineage. The
> WSP *target* has since moved to `slm-125m-e4 → slm-125m-sft`, which already answers QA (0.261
> judged correctness). Its baselines, the train/eval overlap finding and the revised evaluation gaps
> are in Revision 2 §3–§4.

**Target:** `slm-125m` family (v1 base → **v1-extended** → e2 → e4), 125.8M-param LLaMA-style base LM.
**Scope:** read-only audit of what has actually been evaluated, how rigorously, and what remains
untested — specifically in service of a Weight-Space Programming (WSP) experiment on **legal QA**.
**Method:** inspection of the four repos on disk (`pre-training-v1-extended`, `v1-extended-qa-sft`,
`Replicate-the-125M-SLM-Data-Pipeline`, `fine-tuned-125m-slm`), their phase reports, configs, eval
code, and saved metrics. No experiment was run.

> **One-line verdict.** Language-modelling capability is evaluated to a genuinely high standard
> (bits-per-byte, decontamination, bootstrap CIs, modern baselines). **Legal question-answering is
> essentially unevaluated with task-appropriate metrics — because this model does not do QA.** The
> single most important measured fact for the WSP project is that the base model answers
> question-cued retrieval at **0.0% (0/200)** while copying the same spans at **76.5%**. "Improving
> legal QA" on this model means *installing a missing behaviour*, not sharpening a weak one.

---

## 0. What the model actually is (read this first)

Every downstream decision depends on three facts that the WSP brief's framing ("legal-domain
question answering") glosses over:

1. **It is a base completer, not a QA model.** `LlamaForCausalLM`, trained only with next-token
   prediction on US case law / SEC filings / educational web. The model cards state it explicitly:
   *"Not instruction-tuned … it completes text in a legal/financial register; it does not follow
   instructions or answer questions."* (`e2_model_card.md`, `e4_model_card.md`).
2. **No QA fine-tune was ever produced.** The `v1-extended-qa-sft` repo planned a 6-arm SFT study
   but **ran only the Phase 0 capability gate**; its git history ends at
   `1470c1c Phase 0: capability gate`. Arms A–F were never executed. There is no QA checkpoint.
3. **No local checkpoints exist.** Weights live on the HF Hub (`Ace-2504/slm-125m-{base,extended,e2,e4}`)
   and on Modal volumes. Any WSP work must first pull a checkpoint down (125.8M params ≈ 0.5 GB fp32 —
   trivially local-loadable, which is good news for §8).

Architecture (from `config.py`, identical across the lineage): 12 layers · hidden 768 · 12 heads
(head dim 64, MHA) · SwiGLU (inner 3,072) · RoPE θ=10,000 · RMSNorm · **tied embeddings** · vocab
16,384 (byte-level BPE) · context 1,024. Built as a stock HF `LlamaForCausalLM`
(`train_core._build_model`), so `named_parameters()`, forward hooks, and autograd are all available
without custom plumbing — this matters for WSP introspection.

**Tied embeddings is a WSP-relevant detail:** the input embedding and the output (unembedding) head
are the *same* matrix. Any edit to token representations changes both how tokens are read and how
they are predicted. This is a coupling an editor must respect.

---

## 1. Existing evaluation coverage, mapped to the brief's requested dimensions

Legend: ✅ measured rigorously · ⚠️ partial / proxy only · ❌ not measured.

### A. Language-modelling capabilities

| Requested dimension | Status | Where / what |
|---|---|---|
| Training & validation loss | ✅ | Every run logs it; `observations*/metrics*.jsonl`, phase reports |
| Perplexity | ✅ | Val ppl across lineage: base ~11.3 → ext 11.72 → e2 9.86 → e4 9.44 |
| Next-token / token-level accuracy | ❌ | Never reported as accuracy; only cross-entropy loss |
| Generation quality & coherence | ⚠️ | 3 fixed prompts in `eval.py`, qualitative only; no scored rubric |
| Repetition / degeneration | ❌ | Not measured (no rep-n, no distinct-n, no self-BLEU) |
| Context-length sensitivity | ⚠️ | Phase 0 needle probe at ~500-word context only; not swept |
| Performance across prompt lengths | ❌ | Not measured |

**Assessment:** the *density* side of language modelling is evaluated **better than most published
small-model work.** The external `slm-125m-eval` (v4) harness (audited in
`PHASE2A_EVAL_AUDIT.md`) computes **bits-per-byte** — the one metric that is fair across tokenizers —
with a shared UTF-8 denominator, `MAX_LEN=1024/STRIDE=512`, `float32`, `add_special_tokens=False`,
13-gram decontamination against the training corpus, and **document-level bootstrap CIs (1,000
resamples)** against GPT-2, DistilGPT-2, and Pythia-160M. The generation side is thin (3 prompts,
no automatic quality/degeneration metrics).

### B. Legal question-answering capabilities

| Requested dimension | Status | Where / what |
|---|---|---|
| Exact-match accuracy | ❌ | No free-form QA is run; model emits continuations, not answers |
| Semantic similarity | ❌ | Not measured |
| Factual correctness | ❌ | Not measured |
| Legal terminology understanding | ⚠️ | **Proxy only:** bits-per-byte on SEC/case-law domains |
| Legal reasoning / multi-step QA | ⚠️ | **CaseHOLD** 5-way multiple choice, PMI-scored: **0.500** (chance 0.20) |
| Distinguishing similar legal concepts | ⚠️ | Same CaseHOLD signal; not isolated |
| Hallucination rate | ❌ | *Planned* as "fabrication rate" in `SFT_PLAN.md §5`; never run |
| Abstention behaviour | ❌ | *Planned* as "false-abstention rate"; never run |
| Consistency across paraphrases | ❌ | Not measured |

**Assessment:** there is **no direct QA evaluation at all.** The only two legal signals are (i)
compression of legal text (bits-per-byte), which measures familiarity with the *register*, not the
ability to answer, and (ii) **CaseHOLD**, a 5-way multiple-choice holding-selection task scored by
PMI over candidate continuations — the base model sits at **0.500**, above the 0.20 chance floor and
statistically clear of it, but **behind GPT-2 (0.575) and Pythia-160M (0.580)** and tied with
DistilGPT-2 (`PHASE2B_RESULTS.md`). Continued pretraining (v1→extended) changed CaseHOLD by **exactly
0.000**.

### C. General capabilities

| Requested dimension | Status | Where / what |
|---|---|---|
| General knowledge retention | ❌ | Not measured (no closed-book QA / TriviaQA-style probe) |
| Basic reasoning | ⚠️ | Only CaseHOLD (a legal task); no general reasoning benchmark |
| Reading comprehension | ⚠️ | Phase 0 needle retrieval (verbatim-cue 76.5%) is a comprehension proxy |
| Mathematical reasoning | ❌ | Not measured (and near-certainly ~0 at 125M) |
| Instruction following | ⚠️ | **Measured implicitly and it fails:** Phase 0 shows the base continues the passage instead of obeying a question |
| General language generation | ⚠️ | bits-per-byte on C4 / Wikipedia (loses to Pythia) |

### D. Model reliability

| Requested dimension | Status | Where / what |
|---|---|---|
| Response consistency | ❌ | Not measured |
| Sensitivity to prompt variation | ⚠️ | The verbatim-vs-question dissociation (76.5% → 0%) is the sharpest known sensitivity, but phrasing robustness is not swept |
| Output stability across seeds | ❌ | Not measured (decoding is stochastic; no fixed-seed repeat study) |
| Confidence calibration | ❌ | Not measured (no ECE / Brier) |
| Failure modes & systematic errors | ⚠️ | One failure mode is well-characterised: **confident confabulation** under a question cue (`PHASE0_REPORT.md §2.1`) |

---

## 2. The decisive finding: a measured capability dissociation

From `v1-extended-qa-sft/PHASE0_REPORT.md` and `phase0_results.json` (200 paired trials, planted
unguessable codes in ~500-word passages, 0% guessing floor verified by an answer-absent control):

| Probe | Elicitation | v1-extended | 95% CI |
|---|---|---|---|
| Induction (random tokens) | — | 0.880 | [0.872, 0.889] |
| Induction (natural text) | — | 0.837 | [0.821, 0.853] |
| **Needle — verbatim cue** | prompt ends with the fact's own prefix | **0.765** | [0.705, 0.825] |
| **Needle — question cue** | prompt ends with `Question: … Answer:` | **0.000** | [0.0, 0.0] |

**Same fact, same passages, same model. 76.5% when cued by string-continuation, 0/200 when asked a
question.** The retrieval machinery is fully present; the question→span *mapping* is absent. The
failure is not random — the model restates the question as a declarative sentence and confabulates a
fluent, wrong continuation.

**Why this dominates the WSP project:** the target metric ("legal QA accuracy") starts at a hard
**zero**, and the gap to close is a *learned input→output mapping the model has never had*, not a
fact sitting weakly in the weights. This is exactly the regime the brief itself flags as unlikely to
yield to weight editing ("capabilities requiring … sophisticated reasoning"). It must be the
organising constraint of the whole experiment, not a footnote. (It also has an *upside* for
measurement: a metric pinned at 0/200 with a 0% floor has full dynamic range — any real movement is
unambiguous signal, unlike the saturated LLM-judge score that sank the prior SFT study.)

---

## 3. Baseline performance — consolidated real numbers

All figures below are copied from saved artifacts, not re-derived. Scales are **not**
interchangeable (see caveats).

**Validation perplexity (model's own held-out corpus, own tokenizer):**

| base (v1) | extended | e2 | e4 |
|---|---|---|---|
| ~11.3 | 11.72 | 9.86 | 9.44 |

**Bits-per-byte on independent, decontaminated text (lower better) — `PHASE2B_RESULTS.md`:**

| Domain | slm (v1) | slm-extended | gpt2 | distilgpt2 | pythia-160m |
|---|---|---|---|---|---|
| Legal opinions (SCOTUS) | 1.1435 | 1.1500 | 0.9425 | 1.1287 | **0.8277** |
| **SEC filings (LEDGAR)** | **0.6535** | 0.6840 | 0.7711 | 0.9203 | 0.7716 |
| Educational (Wikipedia) | 1.3838 | 1.3700 | 1.0212 | 1.1087 | **0.9521** |
| General web (C4) | 1.3816 | 1.3614 | 1.0797 | 1.1525 | **1.0728** |

→ **SEC is the family's one external win** (beats gpt2 and pythia). Everything else loses to
Pythia-160M (trained on ~147× more data). v1→extended moved **no domain beyond its CI**.

**CaseHOLD (5-way MC, PMI, no BOS, n=200, chance=0.20):**

| slm | slm-extended | gpt2 | distilgpt2 | pythia-160m |
|---|---|---|---|---|
| 0.500 | 0.500 | 0.575 | 0.505 | 0.580 |

**Per-domain forgetting, continued-pretraining legs (loss on v1's held-out probe):**

| Leg | SEC | case-law | web |
|---|---|---|---|
| v1 → extended (515M, 0% SEC) | **+8.12%** | +1.00% | −1.68% |
| e2 (2 ep, 35% SEC) | −12.8% | −6.6% | −4.9% |
| e4 (2 more ep) | further −2.8% | −2.2% | −1.9% |

---

## 4. Identified weaknesses (what the numbers actually say)

1. **Zero question-answering ability** (§2). The headline weakness and the WSP target.
2. **General-register weakness.** Loses to Pythia-160M on legal opinions, educational, and general
   web text; only SEC compression is competitive.
3. **No world knowledge.** 125M params + 1,024 context hold little; the model card says anything
   factual "should come from retrieval, not the weights." Closed-book legal facts are out of reach.
4. **Confident confabulation.** The characteristic failure under a question cue — fluent, wrong,
   ignores the answer sitting in the prompt. This is the behaviour a QA intervention must suppress.
5. **Continued pretraining bought no downstream gain.** +515M tokens moved CaseHOLD by 0.000 and no
   bits-per-byte domain beyond CI — evidence that *more of the same data* is not the lever.

---

## 5. Gaps that block a credible WSP experiment

A WSP loop needs a cheap, reproducible, local, low-variance evaluation it can call hundreds of times.
Today's evaluation assets do not provide that:

| Gap | Why it blocks WSP |
|---|---|
| **No local, one-command eval harness** | `slm-125m-eval` is a separate (off-disk) repo run on CPU over ~120 docs/domain; Phase 0 probes run on Modal. WSP needs an in-process `evaluate(model) → dict` callable with no cloud round-trip. |
| **No primary QA metric wired up** | The deterministic metrics in `SFT_PLAN.md §5` (copy rate, fabrication rate, extractive F1, span recall, false-abstention) are designed but never implemented. WSP has nothing to optimise against. |
| **No frozen eval/edit split** | Editing against the same items you score on is leakage. Needs a held-out QA set + a disjoint "edit-development" set, pre-registered. |
| **No general-capability retention panel** | To measure catastrophic forgetting *relative to the original* (brief Phase 4B), a fixed battery must be snapshotted on the untouched checkpoint before any edit. Only the per-domain loss probe and CaseHOLD exist, and neither is bundled as a reusable "retention panel". |
| **No consistency / calibration instrumentation** | Reliability dimension (D) has zero tooling. |
| **Decoding not pinned for QA** | `eval.py` samples at T=0.8/top-p=0.95 — non-deterministic. Any edit-vs-baseline QA comparison needs greedy, fixed-seed decoding. |

---

## 6. Recommended evaluation framework for the WSP study (prioritised, justified)

Do **not** run every benchmark in the brief. At 125M params, most general benchmarks (MMLU, GSM8K,
etc.) will sit at chance and add noise, not signal. Prioritise metrics that (a) have real dynamic
range on this model and (b) are cheap enough to call in a loop.

**Tier 1 — the WSP target (install the missing mapping).** Build the deterministic QA panel from
`SFT_PLAN.md §5` on a legal-grounded set (RAFT-style: passage + question + extractive answer):

- **Question-cued answer-span recall** — the exact skill at 0%; full dynamic range, 0% floor.
- **Extractive F1** (token overlap vs reference span) — continuous.
- **Fabrication rate** (a number in the answer not present in the passage) — the confabulation
  failure, counted not described.
- **False-abstention rate** — guards against "say *not stated* to everything" collapse.
- **Copy rate / verbatim-cue retrieval (76.5%)** — a *control*: if it degrades, the edit is
  breaking the machinery the QA skill is supposed to build on.

*Why these and not exact-match/semantic-similarity/LLM-judge:* exact-match is too brittle at 125M;
semantic similarity and LLM-judge are expensive and were shown in the prior study to be
unresolvable (96% of answers scored 1/5). Deterministic, bounded, bootstrappable metrics are what
make a closed edit-eval loop trustworthy.

**Tier 2 — catastrophic-forgetting / retention panel (snapshot on the untouched model first).**
Measured as a delta *from the original checkpoint*, with pre-registered regression thresholds:

- **Validation bits-per-byte on SEC / case-law / web** — SEC is the family's only competitive
  advantage and the thing most worth protecting; reuse the v4 harness's fair denominator.
- **CaseHOLD accuracy** (base = 0.500) — the one downstream legal signal already in hand.
- **Per-domain held-out loss probe** — the frozen Phase-0-D protocol, already the family's standard
  forgetting instrument.
- **Verbatim-cue retrieval + induction** — in-context mechanics must survive editing.

**Tier 3 — reliability (cheap additions that close dimension D).**

- **Output stability** — greedy decode is deterministic; measure answer agreement across *paraphrased*
  questions (consistency) and across small weight-edit magnitudes.
- **Confidence calibration** — if abstention is modelled, ECE on answer-token probability; otherwise
  defer (expensive to do well, low priority at this scale).

**Protocol requirements (reproducibility):** fixed seeds; greedy decoding for scored QA; a frozen
eval set *disjoint* from any edit-development set; decontamination of the QA set against the training
corpus (extend the existing 13-gram tooling to `data_cont/corpus`); every number ships with a
bootstrap 95% CI; a result whose CI includes the baseline is reported as **"not resolved,"** never as
a trend. All of this is consistent with the discipline the repo already applies elsewhere.

---

## 7. Caveats (scale hygiene)

- **Three incompatible scales appear above** and must never be mixed: (i) own-corpus perplexity
  (own tokenizer), (ii) independent-text bits-per-byte (tokenizer-fair), (iii) Phase-0-D protocol
  absolute losses. Within each, deltas are valid; across them, absolutes are not.
- **CaseHOLD numbers are the v4-protocol figures** (0.500); the older v3 report's 15.5% is a scoring
  artifact and is superseded.
- **e2/e4 were never run through the external bits-per-byte / CaseHOLD harness** — their evaluation is
  perplexity + the forgetting probe only. If WSP targets e2 or e4 rather than extended, that external
  baseline must be regenerated first.

---

## 8. Bottom line for Phase 2 (feasibility)

The engineering surface is favourable: a 125M stock-LLaMA model, loadable locally in fp32, with full
introspection, and an unusually rigorous measurement culture already in the repo to inherit. The
*scientific* surface is hard: the chosen target (legal QA) is at **absolute zero** and represents a
**missing skill**, which is the least promising case for weight editing. The honest WSP question is
therefore not "can we raise legal QA accuracy" but **"can targeted weight-space edits install any of
the question→answer mapping that the model has never learned, and if so, how far — without
destroying the SEC compression edge and in-context copy mechanics that already work?"** That is a
real, falsifiable question with a measured zero point. Deliverables 2–4 build on it.

---

## 🛑 DELIVERABLE 1 ENDS HERE

No experiment was run. Numbers are transcribed from saved artifacts. The evaluation-framework
recommendation in §6 is the input to the experimental plan (Deliverable 3).

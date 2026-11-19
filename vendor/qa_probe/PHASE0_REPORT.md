# Phase 0 Report — Capability Gate

**Run:** `slm-125m-phase0-probe` on Modal L4 · 2026-07-20 · **cost ≈ $0.20**
**Question:** can v1-extended copy information out of its context at all?
**Verdict:** **PROCEED** — and the probe found something more useful than the gate itself.

All probes are pure continuation. Nothing asks the base model to obey an instruction.
Both models were probed under identical seeds, so every trial is **paired**.

---

## 1. Headline

| Probe | v1 | **v1-extended** | Pre-registered gate |
| --- | --- | --- | --- |
| A · induction, random tokens | 0.841 | **0.880** [0.872, 0.889] | — |
| B · induction, natural text | 0.825 | **0.837** [0.821, 0.853] | — |
| **C · needle retrieval, verbatim cue** | 0.655 | **0.765** [0.705, 0.825] | ≥0.60 → **PROCEED ✅** |
| C · control (needle absent) | 0.000 | **0.000** | guessing floor |
| **C2 · needle retrieval, question cue** | **0.000** | **0.000** [0.0, 0.0] | — |
| C2 · control | 0.000 | 0.000 | |

Copy rate **76.5%** against a **0.0%** guessing floor. The gate threshold (≥60%) is cleared
with the confidence interval entirely above it. **The model can copy from context.**

---

## 2. The actual finding — a total dissociation

> **Verbatim cue: 76.5%. Question cue: 0.0%. Same planted fact, same passages, same models.
> 0 hits in 200 trials, on both models.**

The two probes differ only in how the answer is elicited:

- **Verbatim cue** — the prompt ends with the planted sentence's own prefix
  (`"The internal reference number for this filing is"`). Literal string repeat.
- **Question cue** — the prompt ends with
  `"Question: What is the internal reference number for this filing?\nAnswer:"`.
  Same information, same content words, but the model must map a *question* to a *span*.

**The retrieval machinery is fully present. The question→span mapping is completely absent.**

This is not a capacity ceiling. It is a missing behaviour — which is exactly the kind of
thing SFT exists to install.

### 2.1 The failure mode is diagnostic

The question-cue generations are not random. The model restates the question as a
declarative sentence and then confabulates a plausible continuation:

| Wanted | v1-extended produced |
| --- | --- |
| `KV-6568` | *"The internal reference number for this filing is the same a…"* |
| `KZ-9525` | *"The docket identifier assigned to this matter is the docket…"* |
| `ZZ-6123` | *"The archive catalogue code is available for download at the…"* |

It has the *form* of an answer and none of the *content*. **This is precisely the v2
study's round-2 failure** — "confident, fluent, fabricated figures while the real numbers
sat unused in the prompt" — now reproduced mechanically, in the base model, before any
fine-tuning at all.

---

## 3. This overturns the previous study's conclusion

The v1→v2 QA-SFT study ended on the hypothesis: *"the limit is capacity, not task design.
A 125M model can't learn to use context better than it already does untrained."*

**That is now falsified for the copy mechanism.** v1 — the exact base model that study
fine-tuned — retrieves a planted, unguessable code from a ~500-word context **65.5%** of
the time, with 84% induction accuracy. The machinery was there the whole time.

What was missing, and remains missing at 0/200, is the mapping from a question to the
span that answers it. So the correct restatement is:

> Not *"the model cannot use context."*
> But *"the model can retrieve from context, and has never learned to do so when asked a
> question."*

That is a narrower, more actionable, and more optimistic claim — and it is measured, not
inferred from a null.

### 3.1 A concrete hypothesis for why v2's SFT failed to install the mapping

v2 trained 85% grounded examples for 4,000 pairs and moved nothing. The dissociation
suggests a specific cause worth testing: **v2's teacher-generated answers were abstractive**
— fluent Gemini prose *about* the passage rather than spans *from* it. If so, the training
signal taught "produce a fluent answer of roughly this shape," which is exactly the
behaviour observed (fluent fabrication), and never connected to the copy machinery the
model already had.

**Design implication for this study (§6 below): make a substantial share of SFT answers
extractive — verbatim spans from the passage** — so training targets a capability the model
demonstrably possesses instead of one it does not.

---

## 4. Continued pretraining *improved* in-context retrieval

| | v1 | v1-extended | Δ |
| --- | --- | --- | --- |
| Verbatim-cue retrieval | 0.655 | 0.765 | **+11.0 pp** |

Paired 95% CI **[+5.5, +16.5]** · McNemar exact **p = 0.00031** · discordant pairs 29 vs 7.

**Significant, and previously unmeasured.** The Phase 1 report characterised the
continuation purely as a loss trade (web gain vs SEC loss). It also bought a real,
measurable gain in in-context retrieval that per-domain loss could never have revealed.
Worth adding to the v1-extended write-up.

---

## 5. Per-domain baseline (probe D) — replicates Phase 1

| Domain | v1 | v1-extended | Δ this probe | Δ Phase 1 report |
| --- | --- | --- | --- | --- |
| SEC | 1.9315 | 2.0783 | **+7.60%** | +8.12% |
| case-law | 2.4561 | 2.4857 | **+1.21%** | +1.00% |
| fineweb-edu | 3.3845 | 3.3280 | **−1.67%** | −1.68% |

**Independent replication under a different eval protocol** (batch 8 × 40 batches vs Phase
1's in-training eval). Absolute losses differ — a different subset of val windows — but
every delta reproduces. The forgetting result is robust to eval protocol.

⚠️ **Absolute values here are protocol-specific and are NOT comparable to the Phase 1
report's absolutes.** This probe's protocol is now the frozen zero point for every
forgetting number in this study. Do not mix the two scales.

---

## 6. Consequences for the plan

| # | Change | Reason |
| --- | --- | --- |
| 1 | **Add extractive-answer share to the dataset spec** — a large fraction of grounded answers must be verbatim spans, not teacher paraphrase | §3.1: connects training to machinery the model has |
| 2 | **Adopt question-cue retrieval (C2) as a primary tracked metric** | It is the exact skill SFT must install, it starts at a clean 0.0%, and it has full dynamic range — the opposite of the v2 judge |
| 3 | **Keep verbatim-cue retrieval (C) as a capability control** | If C degrades during SFT, we are damaging the mechanism rather than building on it |
| 4 | Probe D protocol frozen as the forgetting zero point | §5 |
| 5 | Per-item results persisted for every metric from here on | Already implemented in the probe |

Metric 2 is the important one. The previous study's fatal flaw was a metric with no dynamic
range (96% of answers scored 1/5). C2 begins at exactly 0/200 with a 0.0% guessing floor —
**any** movement is real signal, and the ceiling is 100%.

---

## 7. What Phase 0 cost and what it bought

~$0.20 and two Modal runs. It produced: a passed gate, a falsified conclusion from the
previous study, a significant new result about continued pretraining, an independent
replication of Phase 1, a frozen baseline protocol, and a redesign of the dataset spec.

The previous study ran five rounds and ~$1.50 of training before it had a comparably
useful statement about what its model could do.

---

## 🛑 PHASE 0 ENDS HERE

**Gate: PROCEED.** Awaiting green flag for dataset build (8,000 pairs + 500 eval, ≈$5.40
teacher spend), with the extractive-answer change from §6.1 folded into the spec.

**Artifacts:** `phase0_results.json` (full per-item data) · `modal_phase0_probe.py`

# Deliverable 3 — Experimental Research Plan

> **Revised — see [`05_LINEAGE_REVISION.md`](05_LINEAGE_REVISION.md).** The research questions,
> hypotheses (H1–H6), the R0–R6 ladder, the self-recursion design (§6) and the ablations (§7) still
> stand. **Superseded:** the arm list (§4; replaced by Part I "sharpen" S0–S8 and Part II "install"),
> the regression thresholds (§5; to be recalibrated from Gate 1 noise floors) and the data splits
> (§8; replaced by Revision 2 §6).

**Project:** Weight-Space Programming (WSP) for legal QA on the 125M `slm-125m` family.
**Status:** proposal. Nothing is executed. This plan is pre-registered *before* any edit, in the
repo's existing green-flag tradition.
**Depends on:** Deliverable 1 (audit) and Deliverable 2 (feasibility). Read those first.

> **The reframing that governs everything below.** The audit established that legal QA on this base
> model is at an absolute **0/200** on question-cued retrieval, while verbatim copy of the same spans
> is **76.5%**. The target is therefore a **missing input→output mapping**, not weak knowledge. Weight
> editing is weakest exactly here. So the experiment is designed to produce a *credible answer either
> way* — including the honest result "WSP does not beat fine-tuning, and here is precisely how far it
> got and what it cost" — rather than to manufacture a success.

---

## 1. Research questions (the brief's seven, made operational)

| # | Question | Made measurable as |
|---|---|---|
| RQ1 | Can LLM-guided weight edits improve legal QA in a 125M model? | Δ question-cued answer-span recall & extractive F1 vs the untouched base, with bootstrap CIs |
| RQ2 | Are targeted edits better than random/unconstrained ones? | WSP arm vs **magnitude-matched random-edit** and **unconstrained-gradient** arms |
| RQ3 | Can legal gains be had without catastrophic forgetting? | Joint improvement on QA **and** retention panel vs pre-registered regression thresholds |
| RQ4 | Does iterative editing produce *cumulative* improvement? | Monotone, CI-separated gains across iterations on a rotated held-out set (§6) |
| RQ5 | Is there a relationship between edit magnitude and performance? | QA & retention vs ‖Δθ‖ sweep (L2 / relative Frobenius per module) |
| RQ6 | Does WSP beat or complement conventional fine-tuning? | WSP vs **LoRA-SFT** and **full-SFT** on identical data/eval; plus a hybrid arm |
| RQ7 | What are the limits of this paradigm at 125M? | The envelope of all of the above: where each method plateaus and why |

---

## 2. Hypotheses and *pre-registered* expected outcomes

These are predictions committed in advance so the write-up cannot be rationalised post-hoc. They are
deliberately unflattering to the headline idea where the evidence points that way.

| ID | Hypothesis | Predicted outcome (committed now) | Confidence |
|---|---|---|---|
| H1 | Targeted edits install *some* question→span mapping | Answer-span recall rises from 0% to a **small but CI-positive** level (plausibly low single digits to ~low double digits); **not** to parity with a trained QA model | medium |
| H2 | Targeted > random (RQ2) | WSP and gradient arms clearly beat magnitude-matched random edits; random edits mostly **degrade** | **high** |
| H3 | WSP vs fine-tuning (RQ6) | **LoRA-SFT ≥ WSP** on raw QA accuracy at equal or lower cost — because installing an input→output mapping is what gradient SFT is built for. WSP's defensible edge, if any, is in **forgetting control and interpretability**, not accuracy | medium-high |
| H4 | Forgetting is controllable (RQ3) | Small, targeted, reversible edits + replay/regularisation keep SEC bits-per-byte and verbatim-copy within threshold; broad/large edits break them first | medium |
| H5 | Magnitude curve (RQ5) | **Non-monotone:** too small = no effect, too large = degradation/forgetting; a narrow sweet spot | medium |
| H6 | Self-recursion (RQ4) | Early gains, then **plateau**, then risk of degradation; genuine *cumulative* self-improvement is **uncertain** and easily confused with eval overfitting — which §6 is designed to catch | low-medium |

**Explicitly not predicted / not promised:** no 99% accuracy; no acquisition of new world knowledge;
no general-reasoning or math gains; no closed-book legal fact recall. Those are out of scope for
weight editing at 125M and saying otherwise would be dishonest.

---

## 3. What "Weight-Space Programming" means here (so it is not just fine-tuning)

The brief warns against confusing WSP with conventional fine-tuning. We fix a precise operational
definition as a **ladder of increasing "WSP-ness"**, so the result locates *where on the ladder the
value (if any) lives*:

| Rung | Method | Who decides **what** changes | Who decides **how much** | Is it WSP? |
|---|---|---|---|---|
| R0 | Full-parameter SFT | everything (uniform) | gradient descent | **Baseline — not WSP** |
| R1 | LoRA-SFT | fixed low-rank adapters, all layers | gradient descent | **Baseline — not WSP** |
| R2 | Random edit (control) | random modules | random, magnitude-matched | control |
| R3 | Unconstrained gradient edit | all params | one gradient step | control |
| R4 | **Introspection-targeted gradient edit** | **editor LLM selects layers/modules/neurons** from activation & gradient evidence; gradient fills in the values *within that mask* | gradient, bounded | **WSP (defensible core)** |
| R5 | **LLM-guided search over a low-rank edit subspace** | editor LLM proposes candidate directions/targets from eval-failure analysis; gradient-free (ES/CMA-ES) or gradient scoring selects | search | **WSP (emerging)** |
| R6 | **LLM proposes numeric deltas directly** | editor LLM emits the numbers | editor LLM | **WSP (speculative — see Deliverable 2; likely near-hopeless, run only as a documented negative probe)** |

The scientific payload is **R4 and R5**, benchmarked honestly against **R0/R1** (the fine-tuning bar)
and **R2/R3** (the "is targeting doing anything" controls). R6 is included only as a small,
cheap, pre-registered probe so the "can an LLM just write the weights" question is *answered with
data* rather than asserted.

---

## 4. Comparison ladder / baselines (the heart of the design)

All arms: **initialised from the identical untouched checkpoint**, same frozen QA train/edit set,
same frozen eval set, same decoding (greedy, fixed seed), same metrics, same bootstrap matrix.

1. **A0 — Untouched base.** True zero for every metric.
2. **A1 — Random edit** (R2), magnitude-matched to the WSP arm.
3. **A2 — Unconstrained single gradient step** (R3).
4. **A3 — LoRA-SFT** (R1) — the honest, cheap, strong baseline WSP must be compared to.
5. **A4 — Full-parameter SFT** (R0) — upper reference for "what gradient can install here."
6. **A5 — WSP / introspection-targeted edit** (R4) — the core method.
7. **A6 — WSP / LLM-guided subspace search** (R5).
8. **A7 — Hybrid** — WSP targeting to select *where*, LoRA to fill *how* (tests complementarity, RQ6).
9. **A8 — R6 direct-numeric probe** (small, cheap, expected-negative).

**The decisive comparisons:** A5/A6 vs A1/A2 (does targeting matter — RQ2), and A5/A6 vs A3 (does WSP
beat fine-tuning — RQ6). Everything else is context.

---

## 5. Metrics (from Deliverable 1 §6) and success/failure criteria

**Primary (QA target):** question-cued answer-span recall, extractive F1, fabrication rate,
false-abstention rate. **Control:** verbatim-cue retrieval (76.5%) and induction — must not collapse.
**Retention panel (forgetting, measured vs the *original*, not the latest iterate):** SEC/case-law/web
bits-per-byte, CaseHOLD (0.500), Phase-0-D per-domain loss. **Reliability:** paraphrase consistency,
output stability. Every number ships a bootstrap 95% CI.

**Pre-registered decision rules (CI-based, fixed now):**

| Observation | Conclusion |
|---|---|
| A5/A6 QA CI excludes A0 upward **and** excludes A1/A2 | WSP installs real, *targeted* QA signal (RQ1+RQ2 yes) |
| A5/A6 QA CI includes A0 | WSP does not measurably help at 125M — a clean, honest null with error bars |
| A3 (LoRA) QA ≥ A5/A6 beyond CI | **Fine-tuning wins on accuracy**; report WSP's value (if any) only on forgetting/interpretability |
| A5/A6 retention CI within thresholds **while** A3/A4 breach them | WSP's real advantage is *forgetting control*, even if accuracy ties — a publishable result |
| SEC bits-per-byte or verbatim-copy breaches threshold | Edit is damaging the model's one edge / its core mechanics — **fail the arm**, roll back |
| A8 (direct-numeric) ≈ A1 (random) | LLMs cannot write useful weights blind — documented negative (expected) |

**Regression thresholds (acceptable forgetting), pre-registered:** SEC bits-per-byte ≤ **+2%** of the
original (SEC is the protected edge, so the tightest bound); case-law/web ≤ +5%; verbatim-cue
retrieval ≥ **70%** absolute (down from 76.5%); CaseHOLD ≥ 0.46 (within its own CI of 0.500). Any
breach fails the arm regardless of QA gain — the brief's "no improvement at the cost of catastrophic
forgetting" made concrete.

---

## 6. Self-recursion / iterative-editing study (RQ4 — the brief's Phase 4C)

The hard part is distinguishing **genuine cumulative self-improvement** from **overfitting to a fixed
eval set**. Design:

- **Three disjoint frozen splits:** `edit-dev` (the editor may see failures here and propose against
  it), `iter-val` (rotated — a *different* held-out slice scores each iteration, so the loop cannot
  memorise the scorer), and `final-test` (touched **exactly once**, at the very end).
- **Per-iteration tracking:** answer-span recall, extractive F1, retention panel, **‖Δθ‖ from the
  original** (cumulative weight-space distance), validation loss, and *regression on tasks previously
  improved* (did iteration *k* undo iteration *k−1*'s wins?).
- **Convergence / stopping:** declare plateau when `iter-val` gain per iteration is CI-indistinguishable
  from zero for 2 consecutive iterations; declare degradation if the retention panel breaches threshold
  or previously-won tasks regress. Report whether the editor *itself* can recognise the stopping point
  (a stated RQ) — i.e., does its own proposed-edit confidence/anticipated-gain correlate with realised
  gain.
- **The key test (genuine vs illusory):** improvement on the **rotated `iter-val`** and the
  **once-only `final-test`** must track each other. If `edit-dev` and `iter-val` rise while
  `final-test` does not, the loop was overfitting the scorer — and we report that, because it is the
  single most important negative result this sub-study can produce.

---

## 7. Ablations

- **Magnitude sweep (RQ5):** scale the chosen edit by {0.25, 0.5, 1, 2, 4}× and plot QA vs retention —
  tests H5's predicted non-monotonicity.
- **Locus ablation:** restrict edits to {embedding/unembedding (tied!), attention, MLP, final-norm,
  last-k layers} — which sites carry the question→span mapping?
- **Information ablation (what the editor needs — the brief's Phase 2 question):** give the editor
  {weights only, +activations, +gradients, +eval-failure text} and measure how much each information
  source improves proposal quality. This directly answers "what information does the programming LLM
  need to make meaningful modifications."
- **Replay ablation:** WSP with vs without the §6.1-style cumulative-mix replay from `SFT_PLAN.md` —
  does the repo's measured 6× forgetting-protection transfer to editing?
- **Rank/sparsity ablation (R5):** edit subspace dimension vs gains.

---

## 8. Data

Reuse the frozen, decontaminated, extractive-answer QA set specified in
`v1-extended-qa-sft/SFT_PLAN.md` (grounded RAFT-style pairs, teacher-distilled, extractive answers so
the target is a skill the model's copy machinery can support). Splits: `edit-dev`, `iter-val`
(rotated), `final-test`, plus the OOD set. **Decontaminate the QA passages against both training
corpora** (extend the existing 13-gram tooling to `data_cont/corpus`) — the audit flagged this as the
most dangerous leakage path. Keep the replay pool at the model's own cumulative mix (SEC 33.6 /
case-law 33.1 / web 33.3).

---

## 9. Success / failure criteria for the *project* (not just an arm)

- **Primary success:** a defensible, CI-backed answer to RQ1, RQ2, RQ6 — *whichever way they fall* —
  with the SEC edge and copy mechanics preserved. A rigorous null ("WSP ≤ LoRA, here is the envelope")
  is a success.
- **Secondary success:** a characterised forgetting/interpretability niche for WSP even where accuracy
  ties fine-tuning.
- **Failure of the project (not of WSP):** results that cannot be resolved (CIs too wide), or leakage
  that makes gains uninterpretable — the exact failure mode the prior SFT study suffered and that this
  design spends its effort to avoid.

---

## 🛑 DELIVERABLE 3 ENDS HERE

Pre-registered. No arm runs until the plan is approved and Deliverable 4's safeguards are in place.

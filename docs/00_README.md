# Weight-Space Programming (WSP) for the 125M Legal SLM — Research Proposal

This folder is the **proposal stage** of the WSP project: audit the existing model, establish its
evaluation status, research whether WSP is feasible, and design a rigorous experiment. **No weights
have been modified and no experiment has been run.** Implementation (Deliverable 5) begins only after
this proposal is approved.

## Read in order

| # | File | What it answers |
|---|---|---|
| 1 | [`01_MODEL_EVALUATION_AUDIT.md`](01_MODEL_EVALUATION_AUDIT.md) | What has actually been evaluated, how rigorously, and the decisive gap |
| 2 | [`02_WSP_FEASIBILITY_RESEARCH.md`](02_WSP_FEASIBILITY_RESEARCH.md) | Is WSP real? Cited literature, maturity tiers, the honest verdict |
| 3 | [`03_EXPERIMENTAL_PLAN.md`](03_EXPERIMENTAL_PLAN.md) | Pre-registered research questions, hypotheses, baselines, metrics, success/failure rules |
| 4 | [`04_ARCHITECTURE.md`](04_ARCHITECTURE.md) | The 10-component pipeline, safety layer, build order |
| 5 | [`05_LINEAGE_REVISION.md`](05_LINEAGE_REVISION.md) | **Revision 2 (authoritative on target models):** retarget to `e4 → slm-125m-sft`, baselines, leakage finding, revised arms and splits, Gate 1 criterion |
| 6 | [`06_GATE1_REPORT.md`](06_GATE1_REPORT.md) | Gate 1: the instrument reproduces θ_qa; baselines on every axis; approved guardrails |
| 7 | [`07_GATE2_REPORT.md`](07_GATE2_REPORT.md) | Gate 2: the document-count shortcut in the eval and RAFT; LoRA installs abstention but fails retention |
| 8 | [`08_GATE3_PLAN.md`](08_GATE3_PLAN.md) | Gate 3 pre-registration: deterministic search (3a) before editor-LLM arms (3b) |
| 9 | [`09_GATE3A_REPORT.md`](09_GATE3A_REPORT.md) | Gate 3a: no scaled or masked LoRA edit reaches valid abstention within the retention budget |
| 10 | [`10_GATE3B_ACTIVATION_PROBE.md`](10_GATE3B_ACTIVATION_PROBE.md) | Gate 3b-i: θ_qa encodes answer-absence at layer 5 (AUROC ~0.95), but no better than word overlap |
| 11 | [`11_GATE3B_RANK1_EDIT.md`](11_GATE3B_RANK1_EDIT.md) | Gate 3b-ii: rank-1 read→write edit, negative (safety vetoes, three guardrail fails), but ~5× better forgetting per unit of abstention than LoRA |
| 12 | [`12_GATE3B_NEURON_EDIT.md`](12_GATE3B_NEURON_EDIT.md) | Gate 3b-iii: one repurposed neuron gives abstention +0.73 and passes every guardrail except SEC retention (+1.72%); cause diagnosed: misfires inside ordinary text |
| 13 | [`13_GATE3B_AND_GATE_EDIT.md`](13_GATE3B_AND_GATE_EDIT.md) | Gate 3b-iv: AND-gate neuron (absence × reply start), **qualified positive**. Held-out: abstention +0.74, discrimination +0.67, clean F1 ±0, full retention ≤ +0.011% on every set; 3 vectors changed |
| 14 | [`14_OVERLAP_MATCHED_POOL.md`](14_OVERLAP_MATCHED_POOL.md) | Overlap-matched pool (answer sentence deleted, lexical AUROC 0.50): the AND-gate edit's discrimination falls 0.72 → 0.12 (**overlap-dominated**); LoRA 0.88 → 0.26 (partial); θ_qa's own beyond-overlap signal is weak (probe AUROC 0.64) |
| 15 | [`15_FULL_FINETUNE_ARM.md`](15_FULL_FINETUNE_ARM.md) | Full fine-tuning arm: abstention +0.96 and clean F1 +0.105, but retention +4.9–5.9% (**fails**; lr 3e-5 safety-vetoed); matched-pool D 0.25 (PARTIAL, same as LoRA). The AND-gate edit remains the only arm inside every guardrail |
| 16 | [`16_EDITOR_LLM_ARM.md`](16_EDITOR_LLM_ARM.md) | Editor-LLM arm: three independent agents design weight edits from dev feedback only (budget 18 each). **BEATS the deterministic grid in 3 of 3 runs.** All three converged on the L6 gate plus an early-layer slice of a trained delta. Matched D is 0.18–0.23 (PARTIAL) inside every guardrail. The comparison is confounded by the larger action space. |
| 17 | [`17_RANDOM_SEARCH_CONTROL.md`](17_RANDOM_SEARCH_CONTROL.md) | Same-menu control: seeded uniform random search over the editors' action space, same gateway and budget. **R1 BEATS the deterministic grid** (matched D 0.172 [0.136, 0.207], every guardrail held-out), so the run stopped there: the editor win comes from the larger menu, not the LLM. Prediction was wrong |
| 18 | [`18_SELF_RECURSION.md`](18_SELF_RECURSION.md) | Self-recursion (RQ4): re-apply the AND-gate construction to the edited model, aimed at its own remaining failures, at a fresh layer each iteration. **DEGRADES:** θ₂ (an L7 gate) adds abstention +0.13 and raises matched D 0.122 → 0.170 with no forgetting, but held-out clean F1 lower CI −0.022 misses the −0.02 bar; recursion stopped at θ₁ |
| 19 | [`19_FINAL_FROZEN_EVAL.md`](19_FINAL_FROZEN_EVAL.md) | Final frozen eval (pre-registered): every arm once on the untouched 2,000-item instrument, paired against θ_qa, with a transfer check against held-out gains. θ_qa reproduces byte for byte; every arm keeps 97–107% of its held-out gain. The AND-gate fails final-test clean F1 (lower CI −0.030); editors A–C and random R1 pass every guardrail; LoRA and full FT pass QA but fail retention |
| 20 | [`20_FINAL_REPORT.md`](20_FINAL_REPORT.md) | **Final report across all arms:** what weight-space programming achieved, where it fell short, and the answers to the research questions |

## The findings that govern everything

1. **Two lineages, two regimes.** The base completer `slm-125m-extended` answers question-cued
   retrieval at **0/200** while copying the same spans at **76.5%** (Audit §2). But the family also
   has a QA model, `slm-125m-sft` (full fine-tune of `slm-125m-e4` on 15k grounded pairs), at
   **0.261 judged correctness** on a frozen 500-item instrument. The study now uses that lineage:
   **sharpen** θ_qa (primary), and **install** QA into θ_e4 with θ_qa as a ready-made full-SFT
   reference. (Revision 2 §1–§5.)
2. **The literal WSP idea is unproven.** An external LLM *authoring numeric weight edits* in a closed
   loop has **no peer-reviewed precedent**; the "WSP" results circulating online are non-reproducible
   blog posts citing models that don't exist. The nearest real work (SEAL) still makes the weight
   change by gradient descent. (Research §1.5, headline.)
   *Update, 2026-11-19, after the study:* this was the picture at the proposal stage and is now out of
   date. *Sparks of Weight-Space Programming* (Takehi, Teoh, Zhang and Khattab, MIT CSAIL; blog post,
   2 Nov 2026) reports LLMs reading and editing raw weights directly. Open replications and published
   critiques have followed. This study did not review that work, and its pure-WSP arm is a constructed
   edit rather than an LLM authoring raw weights.
3. **So the defensible experiment is honest benchmarking.** Reframe WSP as *LLM-as-controller over a
   gradient-based PEFT edit*, and measure it against plain **LoRA-SFT** and against
   **magnitude-matched random edits** — the design is built to produce a credible answer *either way*,
   including "WSP does not beat fine-tuning, and here is exactly how far it got and what it cost."
   (Plan §3–§5.) The QA task vector τ = θ_qa − θ_e4 makes a small (~110-coefficient), interpretable
   search space in which an LLM-guided editor is tractable. (Revision 2 §2.)
4. **The training pool partly overlaps the eval set.** 483 of 15k QA-SFT pairs share 13-grams with
   eval passages (36 share ≥50%). They are quarantined from `edit-dev`, and their effect on published
   scores is measured at Gate 1. (Revision 2 §4.)

## What is deliberately preserved from the repo's own discipline

Green-flag hard stops; never overwrite the original checkpoint; decontamination against **both**
corpora; bootstrap CIs on every number; a result whose CI includes the baseline is "not resolved,"
not a trend; protect SEC (the family's one external win) as the tightest regression threshold.

## Status

**Complete.** Deliverables 1–4, Revision 2, and Deliverable 5 (Gates 1–6 plus the final frozen
evaluation) are done. The proposal-stage note at the top of this file describes the folder as it was
before implementation. Start with [`20_FINAL_REPORT.md`](20_FINAL_REPORT.md).

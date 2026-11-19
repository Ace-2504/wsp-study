# Weight-Space Programming for a 125M legal LM — final report

**Question.** Can we *program* a trained model's weights directly, with interpretable, constructed
edits rather than gradient fine-tuning? Can such edits install a behaviour θ_qa does not have
(abstaining when the answer is not in the documents) without forgetting, and do they hold up
against LoRA and full fine-tuning on a frozen test set?

**Short answer.** Partly.
- A hand-constructed edit of **one neuron (3 weight vectors)** installs abstention on held-out and
  final-test data, with essentially **zero forgetting**. That is 500–800 times less forgetting than
  LoRA or full fine-tuning.
- But its signal is mostly word overlap, not answer-absence. On the final test it also costs slightly
  more clean F1 than the approved budget allows.
- The edits that pass every guardrail on the final test all **mix a constructed gate with a small
  slice of a fine-tuning delta**. An LLM editor finds them, and so does uninformed random search over
  the same options.
- Pure weight-space programming came close but did not cross the bar. Hybrid edits did.

All numbers below come from the per-gate reports (docs/06–19). Every gate was pre-registered before
it ran, and every prediction that failed is recorded as failed.

## The model and the setup

- **Model:** θ_qa (`slm-125m-sft`), a 125.8M LLaMA-style model fine-tuned for grounded QA.
- **Baseline behaviour:** it never abstains: 0/500 on the final test's retrieval_failure condition.
- **Splits:**
  - `edit_dev` to build edits;
  - `iter_val_0` to select;
  - `iter_val_1..3` for held-out results (n = 3,378);
  - an overlap-matched pool (docs/14) to check whether the signal goes beyond word overlap;
  - the frozen 2,000-item instrument, touched only at Gate 1 (baseline) and in docs/19.
- **Approved guardrails** (against θ_qa):
  - per-domain retention ≤ +0.5%;
  - clean F1 Δ lower CI ≥ −0.02;
  - verbatim copy ≥ 0.93;
  - no significant rise in fabrication.
- The originals were hash-locked and never modified. Every arm is a separate delta file.

## Results across arms

| Arm | What changes | Held-out abstention | Matched-pool D (beyond overlap) | Worst retention | Final-test abstention | Final-test clean F1 Δ | Inside every guardrail |
|---|---|---|---|---|---|---|---|
| LoRA lr 3e-4 (docs/07) | rank-16 adapters, all layers | +0.954 | 0.263 | **+8.87%** | +0.922 | +0.069 | no (retention) |
| Full FT lr 1e-5 (docs/15) | every tensor | +0.960 | 0.254 | **+5.86%** | +0.932 | +0.068 | no (retention) |
| Deterministic WSP grid: rank-1 / single-neuron (docs/11–12) | 1–2 vectors | — | — | +1.72% (neuron) | — | — | no |
| **AND-gate θ₁** (docs/13) | **1 neuron, 3 vectors** | +0.737 | 0.122 (overlap-dominated) | **+0.011%** | +0.724 | **−0.018 [−0.030, −0.005]** | **no** (final-test clean F1, by 0.010) |
| Self-recursion θ₂ (docs/18) | 2 neurons, 6 vectors | +0.868 | 0.170 (partial) | +0.015% | not run | held-out −0.013 [−0.022, −0.005] | no (held-out clean F1, by 0.002) |
| Editors A / B / C (docs/16) | gate + early-layer fine-tuning slice | +0.93 / +0.90 / +0.85 | 0.23 / 0.18 / 0.23 | +0.35 / +0.33 / +0.43% | +0.91 / +0.87 / +0.87 | +0.00 / +0.06 / +0.01 | **yes** |
| Random search R1 (docs/17) | gate + fine-tuning slice | +0.754 | 0.172 | +0.40% | +0.804 | +0.023 | **yes** |

## What we learned

1. **Constructed edits can install a behaviour with almost no forgetting.**
   - The docs/13 AND-gate repurposes one MLP neuron. It fires only when "the answer looks absent"
     *and* "the reply is starting", and it writes the abstention phrase.
   - Forgetting across five held-out text sets is ≤ +0.011%. LoRA and full FT reach 5.9–8.9% for
     similar abstention.
   - The abstention transferred to the untouched final test almost unchanged (+0.724 against +0.737).
2. **But what it reads is mostly lexical overlap.**
   - On the overlap-matched pool, the edit's discrimination falls from 0.72 to 0.12 (docs/14).
   - θ_qa's own internal "answer present" signal beyond overlap is weak (probe AUROC 0.64). A linear
     read of one layer cannot extract much more.
3. **Recursion helps, but it pays in F1.**
   - Re-applying the construction to the edited model's own remaining failures, at a fresh layer, added
     +0.13 abstention. It also lifted the beyond-overlap signal (D 0.122 → 0.170) with no forgetting.
   - The new detector also fires on about 20% of answerable items. Held-out clean F1 then missed the bar
     by 0.002, and the pre-registered stop ended the recursion (docs/18).
4. **The guardrail-passing edits are hybrids, and the LLM was not what made them work.**
   - Three LLM editors, each given 18 evaluations and dev feedback only, all beat the deterministic
     grid. They combined the gate with a small slice of a fine-tuning delta (docs/16).
   - One uninformed random-search run over the same options also beat it (docs/17).
   - The larger set of options mattered, not the LLM's judgement. Whether the editors are *somewhat*
     better than random search is not resolved (one random run).
5. **No arm overfit the iteration loop.**
   - Every arm's final-test abstention gain is 97–107% of its held-out gain. The ranking of arms is
     preserved (Kendall τ 0.90), and θ_qa reproduces byte for byte.
   - The protocol worked as a measuring instrument.
6. **Costs that were not targeted.**
   - False abstention on answerable items rises in every arm: +0.06 (AND-gate) to +0.16 (full FT) on
     the final test.
   - Closed-book F1 falls by 0.05–0.11 in every arm, and most arms also abstain more there.

## Answers to the research questions (docs/03)

- **Can an LLM author useful numeric weight edits in a closed loop?**
  - Yes, in this action space: all three editors produced edits that pass every guardrail on held-out
    data and on the frozen final test.
  - No, as evidence that the LLM matters: random search over the same options did so too.
- **Does WSP beat fine-tuning?**
  - Not on raw abstention: fine-tuning reaches 0.92–0.93 on the final test, against 0.72–0.91.
  - **Yes on the abstention-to-forgetting trade-off.** No fine-tuning arm stays inside the retention
    budget. The AND-gate and the hybrids do.
- **Does it beat magnitude-matched random edits?** Yes, every time. Random edits of equal norm give
  zero abstention (docs/07, 13, 16, 17).
- **Does self-recursion compound?** One step did, at a cost to F1 that just exceeded the approved
  budget (docs/18).

## Limitations

- **One model, one behaviour.** Abstention is a simple output, a fixed phrase, which makes it a
  favourable case for constructed edits.
- **Sample sizes:** one random-search run; three editor runs; a single recursion iteration.
- **The editors were Claude-family agents** working from a written brief. Results may differ for other
  editors or briefs.
- **The final-test retrieval_failure condition always has 2 documents.** Our pools decorrelated
  document count, so no arm was selected on that shortcut, but the instrument itself still carries it.
- **Retention and copy** were measured on held-out text, not on the final instrument. Those
  instruments are separate by design.

## Where everything is

- **Reports:** docs/06–19. Each has Part A (pre-registration) and Part B (result).
- **Code:** `src/wsp/`.
- **Runs, deltas, per-item generations and numbers:** `wsp_runs/`.
- **Final-test numbers:** `wsp_runs/final_eval/final_numbers.json`.
- **Originals:** `checkpoints/_original/`, hash-locked and unmodified.

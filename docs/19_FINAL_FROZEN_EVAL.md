# Final frozen evaluation: every arm, once, on the untouched 2,000-item instrument

**Approved 2026-11-15** ("optional then self recursion then final frozen eval"). **Part A was written
before any edited model was run on `final_test`.** The only earlier use of the instrument is Gate 1's
baseline scoring of θ_e4 and θ_qa. No edit, selection or threshold in docs/07–18 looked at it.

## Part A — pre-registration

**Instrument.** `data/eval/eval_messages.jsonl`: 2,000 items, SHA-256 locked (`16e12b23…`), 500 each of
clean, realistic, retrieval_failure (the answer is absent: the model should abstain) and
closed_book. The harness is Gate 1's, unchanged (`wsp.qa_eval`): one item at a time, greedy, 160 new
tokens, 0-shot, the vendor's metric code. θ_qa is re-run in the same session as the paired baseline.

**Arms** (all fixed before this step; `src/wsp/final_eval.py`):

| Arm | What it is | Source |
|---|---|---|
| θ_qa | baseline | locked original |
| AND-gate θ₁ | pure weight-space programming, 3 vectors | docs/13 |
| LoRA lr 3e-4 | fine-tuning, the Gate 2 carried config | docs/07 |
| Full FT lr 1e-5 | fine-tuning | docs/15 |
| Editors A, B, C | LLM-designed gate plus fine-tuning slice | docs/16 |
| Random R1 | random-search gate plus fine-tuning slice | docs/17 |
| Recursion θ_K | pure WSP, recursive; **included only if** docs/18 accepts K ≥ 2 and θ_K passes every held-out guardrail | docs/18 |

**Measures, each a paired bootstrap CI against θ_qa on the same items:**
- **Target:** abstention rate on retrieval_failure (n = 500).
- **Clean F1:** token F1 on clean (n = 500).
- **Realistic F1:** token F1 on realistic (descriptive).
- **Fabrication:** on clean.
- **False abstention:** on clean plus realistic (n = 1,000).
- Per-condition F1, EM, fabrication and abstention for every arm, including closed_book (descriptive).

**Per-arm FINAL PASS:** all three must hold:
1. retrieval_failure abstention Δ lower CI > 0;
2. clean F1 Δ lower CI ≥ −0.02 (the approved guardrail);
3. clean fabrication not significantly higher.

Retention and verbatim copy are not re-measured: they do not use this instrument. The held-out
values already reported stand.

**Transfer (overfitting) check, docs/03 §5.** For each arm, compare the `final_test` abstention gain
with its pooled held-out `iter_val_1..3` gain.
- An arm **does not transfer** if the upper CI of its `final_test` gain is below half its held-out gain.
- Also reported: whether the ordering of arms by abstention gain is preserved.

**Known difference from the held-out folds, stated in advance.** In this instrument every
retrieval_failure item has exactly 2 documents, all similarity-picked distractors. Our item pools
decorrelated document count from answerability (docs/07), so no arm was trained or selected on a
count shortcut. The eval's distractors are also chosen for overlap with the question, as our hard
negatives were.

**Prediction, stated in advance.**
- Every abstention arm transfers, but with smaller gains than held-out.
- **The AND-gate loses the most.** Its signal is overlap-dominated (docs/14), and these distractors
  are high-overlap by construction. I expect a gain of roughly +0.4 to +0.6, against +0.74 held-out.
- The fine-tuning arms and the editor/random arms keep more of their gain.
- Clean F1 passes for every arm except possibly θ₁: fold 1 alone failed in docs/18's re-run.
- Fabrication does not rise for any arm.
- The ordering is preserved: full FT ≈ LoRA ≥ editors ≥ random ≥ AND-gate.

**This is the last use of `final_test`.** Nothing is tuned after it. If a result is surprising, it is
reported as it is.

## Part B — result

**Harness check.** The θ_qa re-run reproduces Gate 1 **byte for byte on all 2,000 items**. Every
difference below is caused by the weights.

- **The recursion arm** is not included: docs/18 ended DEGRADES, with θ₁ as the last good model.
- **Numbers:** `wsp_runs/final_eval/final_numbers.json`, per-item generations alongside.
- **CIs:** paired bootstrap against θ_qa, n = 500 per condition (n = 1,000 for false abstention).

**Final test, against θ_qa:**

| Arm | Abstention on retrieval_failure | Clean F1 Δ | Clean fabrication Δ | False abstention (clean + realistic) | FINAL PASS |
|---|---|---|---|---|---|
| θ_qa (absolute) | 0.000 | F1 0.229 | fab 0.062 | 0.000 | — |
| **AND-gate θ₁** (pure WSP) | +0.724 [0.684, 0.762] | **−0.018 [−0.030, −0.005]** | −0.006 | +0.062 | **no** (clean F1) |
| LoRA lr 3e-4 | +0.922 [0.898, 0.944] | +0.069 [0.043, 0.095] | −0.038 | +0.135 | yes |
| Full FT lr 1e-5 | +0.932 [0.910, 0.954] | +0.068 [0.041, 0.096] | −0.034 | +0.162 | yes |
| Editor A | +0.908 [0.882, 0.932] | +0.002 [−0.019, +0.022] | +0.008 (n.s.) | +0.122 | yes |
| Editor B | +0.872 [0.842, 0.900] | +0.057 [0.035, 0.079] | −0.006 | +0.106 | yes |
| Editor C | +0.874 [0.844, 0.902] | +0.008 [−0.013, +0.029] | −0.002 | +0.126 | yes |
| Random R1 | +0.804 [0.768, 0.838] | +0.023 [0.003, 0.043] | −0.024 | +0.090 | yes |

**With the held-out measures that do not use this instrument** (retention and copy, from earlier gates):

| Arm | Final-test QA guardrails | Worst full retention (held-out) | Inside every guardrail overall |
|---|---|---|---|
| AND-gate θ₁ | fail (clean F1 by 0.010) | +0.011% | **no** |
| LoRA lr 3e-4 | pass | **+8.87%** | no |
| Full FT lr 1e-5 | pass | **+5.86%** | no |
| Editors A / B / C | pass | +0.35 / +0.33 / +0.43% | **yes** |
| Random R1 | pass | +0.40% | **yes** |

**Transfer (overfitting) check.**
- Every arm transfers. The ratio of final-test to held-out abstention gain is 0.97–1.07:

  | Arm | Ratio |
  |---|---|
  | AND-gate | 0.98 |
  | LoRA | 0.97 |
  | Full FT | 0.97 |
  | Editor A | 0.98 |
  | Editor B | 0.97 |
  | Editor C | 1.03 |
  | Random R1 | 1.07 |

- **No arm overfit the iteration folds on abstention.**
- **Ordering:** held-out gives full FT > LoRA > A > B > C > R1 > AND-gate. The final test gives the
  same order except that B and C swap (0.872 against 0.874, a tie within CI). Kendall τ = 0.90.

**Clean F1 for the AND-gate.**
- Held-out it was −0.005 [−0.013, +0.003], a pass. Here it is −0.018 [−0.030, −0.005], so the lower
  CI misses the −0.02 bar.
- Its F1 cost on answerable text is small but real, and it shows up on this instrument's harder
  answerable items. Realistic F1 is also −0.017 [−0.029, −0.006].
- **The pre-registered rule makes this a FAIL.** docs/13 called the edit "clean F1 ±0", and that does
  not hold on the final test.

**Closed-book (descriptive; no rule set).**

| Arm | Abstention | F1 |
|---|---|---|
| θ_qa | 0.000 | 0.154 |
| AND-gate | 0.394 | 0.084 |
| LoRA | 0.418 | 0.081 |
| Full FT | 0.636 | 0.067 |
| Editor A | 0.856 | 0.085 |
| Editor B | 0.712 | 0.066 |
| Editor C | 0.004 | 0.043 |
| Random R1 | 0.000 | 0.107 |

- These questions have no documents and a different system prompt.
- Whether abstaining here is right is a matter of policy: a grounded assistant with no documents
  arguably *should* abstain. The instrument still scores a gold answer.
- Every edit lowers closed-book F1 by 0.05–0.11. Editor C lowers it most (F1 0.043) **without**
  abstaining, so it answers worse.
- No arm was selected on this condition, and I report it as a cost of the edits.

**Prediction check.**
- **Right:**
  - every arm transfers;
  - fabrication does not rise for any arm;
  - the ordering is preserved;
  - θ₁ is the one at risk on clean F1.
- **Wrong:**
  - "The AND-gate loses the most." It kept 98% of its gain, so overlap-heavy distractors did not hurt
    it more here than on our hard-negative pools.
  - "Smaller gains than held-out." The gains are about equal.

`final_test` has now been used. No further tuning against it.

# Self-recursion: does re-applying the WSP procedure to the edited model, on its own remaining failures, keep improving it?

**Approved 2026-11-15** ("optional then self recursion then final frozen eval"). This implements
docs/03 §6 (RQ4, hypothesis H6). **Part A was written before any recursive iteration was built or
evaluated.**

## Part A — pre-registration

**Question.** Start from θ₁, the docs/13 AND-gate edit: pure weight-space programming, with no
gradient-trained delta. Then apply the same construction again to the *edited* model. Each round
targets only the items that model still gets wrong.

- Does that keep improving the model?
- Does any improvement hold on rotated held-out folds and on the overlap-matched pool, without
  forgetting?

**Why θ₁ and not the editor edits.** The editors' edits contain slices of fine-tuning deltas
(docs/16). Recursing on them would mix fine-tuning into the WSP result.

**One recursive iteration** (θ_{k−1} → θ_k, for k = 2, 3, 4; at most 3 recursive iterations). All
construction data comes from `edit_dev`.

1. **Own failures.** θ_{k−1} answers a fixed, seeded 4,000-item subsample of `edit_dev`. This is the
   same subsample every iteration, chosen with seed `C.SEED`. A **failure** is an absent item that
   θ_{k−1} does not abstain on.
2. **Candidate layers.**
   - The layers in {7, 8, 9} whose MLP has not been edited yet.
   - **Why not L6:** the L6 gate in θ₁ already changed `gate_proj` L6 by 0.2 of its norm. A second
     sharp gate there would exceed the safety gate's 0.25 per-tensor cap.
   - **Why L7 and up:** these are the first layers whose MLP input *contains* the earlier gates'
     writes, so each new read is computed on the edited model's own representation. This is the
     recursive part.
   - The first recursive iteration has 3 candidate layers, the next 2, the last 1.
3. **Read.**
   - A weighted no-intercept logistic probe on that layer's MLP input at the last prompt token. Recipe:
     docs/12 `fit_probe`, with λ chosen by AUROC on the docs/12 10% seed slice.
   - **Positives:** the failures. **Negatives:** the subsample's answerable items, plus generic text
     positions from 2,000 subsample items.
   - Absent items θ_{k−1} already abstains on are left out: they are not errors.
4. **Neuron.** The lowest-importance neuron in that layer, measured on θ_{k−1} (docs/12 importance).
5. **Detector r.** The ridge reply-start detector, refit on θ_{k−1} at that layer using the docs/13
   recipe and admissibility cap.
6. **Write δ.** `optimize_write` on θ_{k−1}, trained on the failure items only, with ρ ∈ {0.5, 1}.
7. **Gate.**
   - τ = 0 and gain 1.
   - S is the largest value ≤ 4 with ‖Δ gate row‖ ≤ 0.2‖W_gate‖.
   - m is the median firing activation on failures.
   - The new neuron's rows are added to θ_{k−1}'s delta. The safety gate checks the *cumulative*
     change against θ_qa, with unchanged limits.
8. **Grid and selection on `iter_val_0`.**
   - The grid is candidate layer × ρ.
   - The selection rule is the docs/13 rule: safety pass, guardrails pass, validity, then the largest
     abstention gain.
   - **Additional rule:** the pick must raise the `iter_val_0` abstention gain above θ_{k−1}'s.
   - If no config qualifies, recursion **stops (no improving edit)**.

**Design choices, and their basis.** These differ from docs/13:
- **τ = 0 only:** τ = 1 was worse in docs/13 and in all three editor runs.
- **Layers 7 to 9:** the per-tensor budget, explained above.
- **ρ ∈ {0.5, 1}:** to keep the grid small.

All of these are based on dev results or on construction only.

**Tracking per accepted model θ_k (θ₁ is re-evaluated in the same harness):**
- **Folds:** `iter_val_0..3`.
- **Measures:** abstention Δ, clean F1, false abstention, fabrication, discrimination, fast
  retention, and cumulative ‖Δθ‖/‖θ‖.
- **The held-out overlap-matched pool (docs/14):** D matched with its verdict.
- **Regression on earlier wins** (`iter_val_1..3`): the share of items θ_{k−1} got right (absent →
  abstained; answerable → not abstained) that θ_k now gets wrong.
- **At the end:** full retention on both disjoint sets, and the verbatim-copy probe, for every
  accepted θ_k.

**Rotation and stopping** (docs/03 §6):
- **Scoring fold:** θ_k's scoring fold is `iter_val_{k−1}` (θ₂ → fold 1, θ₃ → fold 2, θ₄ → fold 3).
  On it, the paired abstention Δ of θ_k against θ_{k−1} is reported.
- **Plateau:** that CI includes 0 for 2 consecutive iterations → stop.
- **Degradation:** θ_k fails any approved guardrail held-out, pooled `iter_val_1..3` against θ_qa →
  stop. θ_{k−1} is then the last good model.
- **`final_test` is not touched here.** It is reserved for the final frozen-eval step.

**Outcome**, for the last good model θ_K against θ₁, held-out:
- **GENUINE CUMULATIVE GAIN:** K ≥ 2, θ_K passes every guardrail, and θ_K's matched-pool D lower CI
  is above θ₁'s point estimate. Its matched D is re-measured here; docs/14 found 0.122.
- **ABSTENTION-ONLY GAIN:** K ≥ 2, guardrails pass, and pooled abstention is significantly higher
  than θ₁'s (paired CI > 0), but matched D is not.
- **NO GAIN:** no recursive iteration accepted, or neither of the above.
- **DEGRADES:** the first recursive model already fails a held-out guardrail.

**Prediction, stated in advance (H6: early gain, then plateau): ABSTENTION-ONLY GAIN.**
- One or two iterations are accepted, adding about +0.03 to +0.10 abstention held-out, with false
  abstention rising.
- Matched D does not clearly exceed θ₁'s. The failures left are mostly high-overlap absent items,
  which a linear read on the same features separates poorly.
- The plateau or the per-tensor budget then ends the recursion.

**Limitation, stated in advance.** "Self" means the WSP *procedure* re-applied to the model's own
failures and its own edited representation. A 125M model cannot propose edits itself. The docs/03
question of whether the editor can recognise its own stopping point does not apply: there is no LLM
editor in this loop.

## Part B — result

**Outcome (pre-registered rule): DEGRADES.**
- The first recursive model, θ₂, was accepted on dev.
- It fails a held-out guardrail: pooled clean F1 Δ is −0.013 with CI [−0.022, −0.005], against the
  bar of lower CI ≥ −0.02. It misses by 0.002.
- Recursion stopped after one iteration, and **θ₁ remains the last good model**. My prediction
  (ABSTENTION-ONLY GAIN) was wrong in kind.
- The near-miss matters, so the rest of the picture is reported in full below. Every number is in
  `wsp_runs/gate6_recursion/gate6_recursion_numbers.json`.
- *Engineering note:* the first launch crashed before building iteration 2. It was a dtype bug: the
  abstain flags were floats, and the code inverted them with `~`. Nothing had been selected or
  scored. After the fix, the run resumed from the cached θ₁ evaluations (`run_crash1.log`).

**Iteration 2 (θ₁ → θ₂).**
- **θ₁'s own failures:** on the 4,000-item `edit_dev` subsample, θ₁ abstains on 66.7% of absent
  items, leaving 666 failures. It abstains on 2.6% of answerable items.
- **The read at each new layer:**
  - probe slice AUROC 0.83–0.84 at L7, L8 and L9;
  - the detector fires on 88–92% of the failures, **but also on 19–21% of answerable items**.
- **Grid** (6 configs on `iter_val_0`): all passed safety. Only L7 ρ0.5 and L9 ρ0.5 passed every dev
  guardrail; the ρ1 writes failed dev clean F1.
- **Selected: L7, ρ0.5.** Dev abstention gain was +0.859 against θ₁'s +0.716. Dev clean F1 was
  +0.001, which passed.

**Held-out, pooled `iter_val_1..3` (n = 3,378) against θ_qa:**

| | θ₁ (AND-gate, 3 vectors) | θ₂ (+ L7 gate, 6 vectors) |
|---|---|---|
| ‖Δθ‖/‖θ‖ | 0.0330 | 0.0410 |
| Abstention Δ | +0.737 [0.723, 0.752] | **+0.868** [0.856, 0.879] |
| Discrimination (min fold) | 0.665 | 0.726 |
| False abstention Δ | +0.035 | **+0.108** [0.097, 0.118] |
| Clean F1 Δ | −0.005 [−0.013, +0.003] ✓ | **−0.013 [−0.022, −0.005] ✗** |
| Fabrication Δ | −0.007 | −0.014 |
| Worst full retention | +0.011% | +0.015% |
| Verbatim copy | 0.955 | 0.955 |
| **Matched-pool D** | 0.122 [0.097, 0.150], overlap-dominated | **0.170 [0.136, 0.204]**, PARTIAL |

**Against θ₁:**
- **Scoring fold** (`iter_val_1`, θ₂ against θ₁): abstention +0.135 [0.115, 0.156].
- **Pooled:** abstention +0.131 [0.120, 0.142], false abstention +0.073, clean F1 −0.009
  [−0.015, −0.003].
- **Regression on earlier wins:** 248 of θ₁'s 5,748 correct items (4.3%) are now wrong, while 445
  items θ₁ got wrong are now right.

**Interpretation.**
1. **The recursion did improve what it was aimed at, without forgetting.**
   - Re-reading the *edited* model's representation at L7, one more neuron lifted abstention by
     +0.13 held-out.
   - Matched D rose from 0.122 to 0.170. Its lower CI, 0.136, is above θ₁'s point estimate, which
     would have met the GENUINE CUMULATIVE GAIN bar if the guardrails had held.
   - Retention and copy are unchanged (+0.015% and 0.955). This is the first pure-WSP edit with a
     PARTIAL rather than overlap-dominated matched D.
2. **The cost falls on answerable questions, and the dev fold did not show it.**
   - The new detector fires on about 20% of answerable items. Held-out false abstention tripled and
     clean F1 slipped just past the bar.
   - On dev the selected config looked clean (+0.001 F1). On the held-out folds it is −0.007 to
     −0.021. The selection fold under-estimated the F1 cost: a small instance of the dev-overfitting
     risk this protocol is built to catch.
3. **What the pre-registered rule means here.** The guardrail is binary, and θ₂ fails it.
   - That makes the reported outcome DEGRADES. I am not relabelling it.
   - The descriptive conclusion: **a second recursive step trades about 0.01 clean F1 and +0.07
     false abstention for +0.13 abstention and +0.05 matched D.**
   - Under the approved guardrails, that trade is just outside the budget.

**Not tested** (each would be a new pre-registration, not a rescue of this one):
- choosing the next edit with an F1 margin on dev;
- training the detector with answerable items as explicit negatives;
- continuing from θ₂ despite the stop.

`final_test` was not used. By the docs/19 rule, θ₂ is **not** added to the final frozen eval.

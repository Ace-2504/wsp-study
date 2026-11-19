# Gate 3b-iv — the AND-gate neuron edit

**Approved 2026-11-10** (after the single-neuron edit's near miss, docs/12). **Part A was written before
any new vector is built or any configuration is evaluated.** All construction data comes from
`edit_dev`. No dev-fold, held-out or retention-window data is used to build anything.

## Part A — pre-registration

**What changes, and nothing else.** This is the docs/12 edit with **one vector swapped**. The
diagnosis in docs/12 was:
- the repurposed neuron misfires inside ordinary text (`,` `of` in SEC filings, `.` in case law);
- each misfire writes the abstain push mid-document;
- zeroing the neuron itself costs nothing.

The neuron computes `SiLU(g_jᵀx̂) · (u_jᵀx̂)`. In docs/12 the second factor was a constant
(u_j = c, so u_jᵀx̂ ≈ 1 everywhere). Here it becomes a **reply-start detector** r. The neuron is then a
product, "answer looks absent" **AND** "a reply is starting", which the SwiGLU block already
computes; no new mechanism is added.

Reused unchanged from `wsp_runs/gate3b_neuron/vectors/` (built from `edit_dev` only, deterministic):
- the layer, the neuron j (L5: 3042, L6: 1880), the gate sharpness S, the near-constant direction c;
- the absence probe w (real, shuffled and all-positive variants), so the gate row is unchanged;
- the write vectors δ for ρ ∈ {0.25, 0.5, 1.0}.

**The edit.**
- **gate row g_j = S · (w − τ·c)**, exactly as in docs/12.
- **up row u_j = r**, the reply-start detector (new).
- **down column W_down[:, j] = δ / m**, where m is recomputed as the median of
  `SiLU(S·(wᵀx̂ − τ·cᵀx̂)) · (rᵀx̂)` over `edit_dev` absent items whose gate fires.

**The detector r** is a weighted ridge regression on x̂, the block-ℓ MLP input:

    r = argmin  Σ_groups (1/N_g) Σ_{i∈g} (rᵀx̂_i − y_i)²  +  λ ‖r‖²

Five groups, equal total weight:

| group | positions | target |
|---|---|---|
| reply start | the last prompt token (`\n` after `<|assistant|>`) of **every** `edit_dev` item, absent and answerable | 1 |
| prompt text | every 8th non-final prompt position, 2,000 fixed `edit_dev` items (the docs/12 sample) | 0 |
| template tokens | every special token and the token right after it, except the final position, same 2,000 items | 0 |
| reply tokens | the first 32 teacher-forced gold-reply tokens, same 2,000 items | 0 |
| raw document text | the user message's document text **alone, without the chat template** (as retention windows are fed), first 1,000 tokens, every 4th position, same 2,000 items | 0 |

- The raw-text group is the direct fix: docs/12's negatives never included text outside the template.
- The reply-token group keeps the neuron silent once a reply is under way. The write δ was optimised
  only for the reply-start position, so this matches how δ was built.
- **λ = λ_rel · tr(Σ̄)/768**, with λ_rel ∈ {1e-4, 1e-3, 1e-2, 1e-1}.
  - Admissible λ: the up-row change ‖r − u_j,old‖ ≤ 0.2·‖W_up‖. This is the same cap as the gate row,
    below the 0.25 safety limit.
  - Chosen λ: among admissible values, the lowest weighted held-out MSE on the fixed 10% seed slice
    of `edit_dev` (the docs/11 slice), fitting on the other 90%.
  - r is then refit on all of `edit_dev`.
  - If no λ is admissible, **stop**.

**Construction checks reported in the build log** (descriptive; not selection criteria):
- detector mean, rms and 99th percentile in each group, on the slice;
- the neuron's firing rate on raw document text, set beside the docs/12 neuron on the same positions.

**Grid (dev fold `iter_val_0`, 12 configurations).** Block ℓ ∈ {5, 6} × ρ ∈ {0.25, 0.5, 1.0} ×
τ ∈ {0, 1}. This is identical to docs/12, so every row pairs with a docs/12 row that differs only in
the up row.

**Pipeline, selection and stopping rule.** Unchanged from docs/11 and docs/12:
- the safety gate, the approved guardrails (retention ≤ +0.5% per domain on the fast subset; clean-F1
  lower CI ≥ −0.02; no significant fabrication rise) and the validity condition (discrimination min ≥
  half the abstention gain);
- select the largest abstention gain among the configurations that pass;
- **if none passes, stop**: no held-out evaluation.

**Report for a selected edit.**
- Held-out folds `iter_val_1..3` pooled, paired vs θ_qa.
- 3 random controls, matched per tensor in rank and norm (`random_like`).
- Structured controls, on dev and held-out:
  - **(a) shuffled read:** the shuffled-label absence probe in the gate; r is kept;
  - **(b) write-only:** the all-positive probe in the gate; r is kept, so the neuron fires at every
    reply start;
  - **(c) ablation only:** neuron j zeroed, nothing written.
- **Milestone:** full retention on both disjoint held-out sets, plus the copy and induction probes.

**What counts as a positive result.** On the held-out folds, the selected edit must:
1. pass every approved guardrail, including full retention ≤ +0.5%;
2. pass the validity condition;
3. beat the 3 random controls and controls (a) and (b) on abstention.

**Prediction, stated in advance.**
- Retention should fall well below docs/12's (SEC +1.72% at L6 ρ1 τ0), with abstention roughly held.
- The risk: r may not reach ≈ 1 at the reply start without exceeding the norm cap. A detector weaker
  at the last token gives a smaller m and so a larger down column, which may hit the safety gate. The
  build log will show this before any evaluation.

**Known limit (as in docs/10–12).** On this item pool the absence signal is roughly word overlap.

## Part B — results (`wsp_runs/gate3b_andgate/`: `build.log`, `run.log`, `log.jsonl`, `results.json`, `gate3b_andgate_numbers.json` from `python -m wsp.and_gate_report`)

**Verdict: the first edit in this project to pass every approved guardrail and the validity condition on
the held-out folds.** It beats all three random controls. Against the pre-registered structured controls
the result is **qualified**, as set out under Criterion 3 below, so this is reported as a **qualified
positive**, not a clean pass.

### What was built (from `edit_dev` only)

| | block 5 | block 6 |
|---|---|---|
| chosen λ_rel (all four admissible) | 1e-3 | 1e-2 |
| up-row change ‖Δu_j‖ / ‖W_up‖ | 0.018 | 0.019 |
| detector rᵀx̂ on the 10% slice: reply start, mean (1st percentile) | 0.97 (0.84) | 0.98 (0.83) |
| detector rms: prompt text / template / reply tokens / raw text | 0.086 / 0.060 / 0.076 / 0.092 | 0.081 / 0.059 / 0.070 / 0.087 |
| neuron fires (act > 0.5, τ 0) on raw document text: docs/12 → AND gate | 1.01% → **0.03%** | 1.14% → **0.02%** |
| neuron fires on template tokens: docs/12 → AND gate | 0.42% → 0% | 2.36% → 0.06% |
| median firing activation m (τ 0): docs/12 → AND gate | 1.18 → 1.16 | 3.04 → 2.96 |

The detector works as designed. Misfires on raw text fall about 50×, and m barely moves, so the down
column is almost the same as in docs/12.

### Dev-fold grid (`iter_val_0`), each row beside its docs/12 twin (same edit, constant up row)

| config | abstain Δ | discrimination min | clean F1 Δ | false-abstain Δ | retention sec / case-law / web | docs/12 retention sec | outcome |
|---|---|---|---|---|---|---|---|
| L5 ρ0.25 τ0 | +0.048 | +0.003 | +0.003 | +0.001 | +0.00 / −0.00 / −0.00% | +0.62% | fail: validity (too weak) |
| L5 ρ0.25 τ1 | +0.053 | +0.006 | +0.014 | +0.001 | +0.00 / −0.00 / −0.00% | +0.83% | fail: validity (too weak) |
| L5 ρ0.5 τ0 | +0.500 | +0.383 | −0.005 | +0.030 | +0.02 / +0.00 / +0.00% | +3.77% | fail: clean F1 (lower CI) |
| L5 ρ0.5 τ1, ρ1 τ0, ρ1 τ1 | — | — | — | — | — | — | safety veto (down column), as in docs/12 |
| L6 ρ0.25 τ0 / τ1 | +0.008 / +0.009 | 0.000 | +0.006 / +0.011 | ≈0 | +0.00% everywhere | +0.02 / +0.03% | fail: validity (too weak) |
| L6 ρ0.5 τ0 | +0.524 | +0.449 | −0.004 | +0.025 | +0.01 / −0.00 / +0.00% | +1.44% | **pass** |
| L6 ρ0.5 τ1 | +0.468 | +0.413 | +0.004 | +0.018 | +0.01 / +0.00 / +0.00% | +1.72% | **pass** |
| **L6 ρ1 τ0** | **+0.716** | **+0.633** | +0.007 | +0.066 | **+0.01 / +0.00 / +0.00%** | +1.72% | **pass → selected** |
| L6 ρ1 τ1 | — | — | — | — | — | — | safety veto (tensor 0.296), as in docs/12 |

- The only change from docs/12 is the up row. Abstention is unchanged to within ±0.02 in every config.
- Fast-subset retention falls from +0.6–3.8% to **+0.00–0.02%**.
- The misfire diagnosis of docs/12 is confirmed: the misfires were essentially all of the forgetting.

### Held-out folds `iter_val_1..3` pooled (1,126 items per kind; never used for any choice), paired vs θ_qa

| arm | abstain Δ, abs1–3 | abstain rate abs1 / abs2 / abs3 | false abstain ans1 / ans2 / ans3 | discrimination min | clean F1 Δ (ans1) | clean fabrication Δ | guardrails | validity |
|---|---|---|---|---|---|---|---|---|
| θ_qa | — | 0.000 / 0.000 / 0.000 | 0.001 / 0.000 / 0.000 | — | — | — | — | — |
| **selected (L6 ρ1 τ0)** | **+0.737 [+0.723, +0.752]** | 0.756 / 0.745 / 0.710 | 0.025 / 0.036 / 0.045 | **+0.665** | −0.005 [−0.013, +0.003] | −0.007 [−0.016, +0.001] | **PASS** | **PASS** (0.665 ≥ 0.369) |
| random s0 / s1 / s2 | +0.000 | 0 / 0 / 0 | ≈0 | ≈0 | −0.005 / −0.001 / +0.010 | n.s. | pass | — |
| (a) shuffled read | — | — | — | — | — | — | **safety veto** (tensor 1.28) | — |
| (b) write-only | +0.974 [+0.969, +0.979] | 0.980 / 0.979 / 0.964 | 0.980 / 0.977 / 0.955 | **0.000** | **−0.174** | −0.074 | FAIL (clean F1) | FAIL |
| (c) ablation only | +0.000 | 0 / 0 / 0 | ≈0 | ≈0 | +0.002 | +0.000 | pass | — |

- Across ans1–3, false abstention is +0.035 [+0.029, +0.041] and F1 is −0.007 [−0.011, −0.002].
- **Per fold, not pooled** (pre-registered as pooled; disclosed for completeness): fold 1's clean-F1
  lower CI falls below −0.02 (Δ −0.009), while folds 2 and 3 pass. The pooled CI is the
  pre-registered test, and it passes with room to spare.
- Abstention output: 2,575 of 2,610 abstentions are exactly `not stated in the context`. The rest
  are near-variants ("…in the context.", "…of the context").
- Answer stability: 236 of 1,126 clean (ans1) answers changed text vs θ_qa (21%), with no net F1
  loss. For the rank-1 edit it was 61%.

### Milestone: full retention on both disjoint sets (all windows), vs θ_qa

| arm | e4_native sec | e4_native case-law | e4_native web | v1_heldout case-law | v1_heldout web | ≤ +0.5%? | verbatim copy (≥ 0.93) |
|---|---|---|---|---|---|---|---|
| **selected** | **+0.011%** | **+0.006%** | **+0.002%** | **+0.008%** | **+0.003%** | **PASS** | **0.955 PASS** |
| random s0 (same norm) | +0.57% | +0.33% | +0.23% | +0.35% | +0.23% | FAIL | 0.960 |
| (c) ablation only | +0.000% | +0.001% | +0.001% | +0.002% | +0.000% | pass | 0.960 |

- The selected edit's increases are statistically detectable (paired CIs exclude 0) but about 50×
  inside the budget.
- They are **≥ 50× smaller than a random edit of the same norm** and **≥ 700× smaller than the Gate 2
  LoRA** (+7.5–8.8%).
- The induction probes are unchanged: random-token gain 0.821 vs 0.819 (random s0); natural-text
  gain 0.408 vs 0.406.
- C2 needle-question is 0.045 (θ_qa ≈ 0.05).

### Against the pre-registered positive-result criteria

1. **Every approved guardrail on held-out data, including full retention ≤ +0.5%: PASS.**
2. **Validity (discrimination min ≥ half the gain): PASS** (+0.665 vs 0.369, at every document
   count).
3. **Beat the 3 random controls and controls (a) and (b) on abstention: met for the random controls
   (+0.737 each, CI [+0.723, +0.752]); NOT met as literally worded for (b); not assessable for (a).**
   - **(b) write-only:** it abstains on 97% of absent items, more than the selected edit's 74%
     (paired −0.237), because it abstains on 97% of *answerable* items too. Its discrimination is
     0.000 vs the selected edit's +0.665; it fails clean F1 and validity.
     - The purpose the pre-registration gave for (b) was to separate "routes the absence signal"
       from "abstains more everywhere". On that purpose the selected edit wins decisively.
     - But the wording says "on abstention", and on raw abstention it loses.
     - The same wording appears in docs/11 and docs/12. It is a flaw in the criterion, found only
       now; **it is not being amended after the fact.**
   - **(a) shuffled read:** vetoed by the safety gate. The shuffled probe almost never fires (m =
     0.49 at L6), so δ/m makes the down column 1.28× the matrix norm. The control therefore could not
     be evaluated. The docs/12 shuffled control had the same construction.

**Hence "qualified positive".** On held-out data the edit:
- installs answer-absence abstention that is real by every pre-registered test of discrimination and
  of matched random controls;
- costs no measurable answering quality;
- forgets ≤ 0.011% on every retention set;
- changes 3 vectors (2,304 numbers) of a 125.8M-parameter model.

What it does not do: satisfy a control criterion whose wording did not anticipate that a "fire
everywhere" control would abstain more.

### What this is and is not

- **It is a weight edit, not fine-tuning.**
  - No weight was trained by gradient descent on a loss over the data.
  - Three weight vectors were **written** from closed-form fits on θ_qa's own activations: a logistic
    probe, a ridge detector and a near-constant direction.
  - One 768-d activation vector δ was optimised by gradient, then written into a weight column.
- **It is not yet the literal WSP idea.** No external LLM authored these numbers; I designed the
  construction, and every number came from fits to `edit_dev`. The editor-LLM arm (plan step 2) is
  still to run.
- **Compared with LoRA** on the same held-out folds:
  - LoRA abstains more (0.94–0.96 vs 0.71–0.76);
  - it has higher false abstention (8.6% vs 3.5%);
  - it gives clean F1 +0.10, where this edit gives ±0;
  - its forgetting fails the budget about 15–17× over, where this edit is about 50× inside it.
- **Known limit (docs/10), unchanged.** On this pool the absence signal is about as good as word
  overlap. This edit may implement "abstain when the documents don't share the question's words".
  The **overlap-matched control pool** is the test, and it is now the most important next step.

### What this unlocks in the plan

An edit has passed the guardrails, so the pre-registered follow-ups apply:
- **the self-recursion study:** can the edited model propose or improve its own next edit?
- **the overlap-matched pool**, to test what the installed skill actually is;
- **the comparisons:** the editor-LLM arm and the full-SFT comparison.

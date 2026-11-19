# Gate 3b-ii — a rank-1 read→write edit for abstention

**Approved 2026-11-10** (after the probe came back PRESENT, docs/10). **Part A is written before any
edit is built or evaluated.** Part B follows the run.

## Part A — pre-registration

**Idea.** θ_qa already computes "the answer is absent" (layer-5 probe AUROC ≈ 0.95) but never acts on
it. The edit routes that existing signal to the abstain response with **one rank-1 change to one weight
matrix**. It is a closed-form edit in the ROME family (Meng et al., 2022), not fine-tuning:
- no weight is trained by gradient descent;
- the edit is an outer product of two vectors (3,072 + 768 numbers);
- both vectors are computed from θ_qa's own activations.

**The edit.** For block ℓ:

    W_down(ℓ) ← W_down(ℓ) + γ · δ_ℓ r_ℓᵀ        (rank 1; one matrix; nothing else touched)

`W_down` maps the 3,072-d MLP hidden activation h to the 768-d residual stream. After the edit, every
position gets the extra term γ·δ·(rᵀh). The two vectors do different jobs:

- **Read vector r_ℓ** (3,072-d, closed form) should give rᵀh ≈ 1 at the last prompt token of
  answer-absent items, and ≈ 0 at answerable items and at ordinary text positions:

      r = argmin (1/N) Σ_last-token (rᵀh − y)²  +  w · E_generic[(rᵀh)²]  +  λ‖r‖²
      y = 1 (absent), 0 (answerable);  h = input of W_down(ℓ)

  - **Data:** last-token keys from all 12,732 `edit_dev` items.
  - **Generic positions:** every 4th non-final prompt position of a fixed sample of 2,000 `edit_dev`
    items. These are text positions where the edit should stay silent. They are disjoint from the
    retention evaluation windows.
  - **Weights:** w = 1, and λ = λ_rel · tr(H̄)/3072 with λ_rel ∈ {1e-4, 1e-3, 1e-2, 1e-1}.
  - **Choosing λ:** by AUROC on a 10% seed-level slice of `edit_dev`, fitting on the other 90%. The
    vector is then refit on all of `edit_dev`.
- **Write vector δ_ℓ** (768-d) is the vector which, added to block ℓ's output at the last prompt token,
  maximises the mean teacher-forced log P("not stated in the context<|eos|>") on `edit_dev`
  answer-absent items. θ_qa is frozen. Only this 768-d activation vector is optimised: Adam,
  300 steps, batch 16, seed 1337.
- **The knob γ** scales the whole edit.

**Grid (dev fold `iter_val_0`).** Block ℓ ∈ {5, 6, 7, 8}, the blocks whose input already carries the
signal (probe ≥ 0.92 at hidden states 5–8), and γ ∈ {0.5, 1, 2}. That is **12 configurations**, fewer
than Gate 3a's 18.

**Pipeline.** Identical to Gate 3a:
- the safety gate first;
- dev-fold metrics paired vs θ_qa;
- the approved guardrails, unchanged;
- the validity condition: discrimination min over k ≥ half the abstention gain.

**Selection.** The largest abstention gain among configurations that pass safety, the guardrails and
validity. **If none passes, stop**: report the grid, with no held-out evaluation.

**Report for a selected edit.**
- Held-out folds `iter_val_1..3` pooled, paired vs θ_qa.
- **3 random rank-1 controls** with the same matrix, the same norm, and random r and δ.
- **2 structured controls**, run on the dev fold and on the held-out folds:
  - **(a) shuffled-read:** r is refit on shuffled labels and δ is kept;
  - **(b) write-only:** δ is added through a read vector that fires equally on absent and answerable
    items (fit on y = 1 for all items).

  These separate "routes the absence signal" from "abstains more everywhere".
- **Milestone:** full retention on both disjoint held-out sets, plus the copy and induction probes.
- **The comparison:** against the Gate 2 LoRA, both its abstention and its forgetting.

**What counts as a positive result.** On the held-out folds, the selected edit must:
1. pass every approved guardrail, including full retention ≤ +0.5%;
2. pass the validity condition;
3. beat all three random controls **and** both structured controls on abstention.

Anything less is reported as negative or partial, with numbers.

**Known limit, stated in advance.** On this pool the absence signal is about as good as word overlap
(docs/10). An edit that passes therefore implements roughly "abstain when the documents don't share
the question's words". The overlap-matched control pool would test that, and it is the next step
whatever the outcome.

## Part B — results (`wsp_runs/gate3b_rank1/`: `build.log`, `run.log`, `results.json`, `vectors/build_meta.json`)

**Verdict under the pre-registered rule: NEGATIVE. No configuration passed.**
- **11 of 12 were vetoed by the safety gate** before evaluation. The rank-1 change to one matrix was
  0.25–1.3× that matrix's own norm, over the 0.25 per-tensor limit.
- **The one configuration that ran (L5, γ = 0.5)** produced real, document-count-independent
  abstention (+0.19). But it **failed three guardrails**: clean F1, fabrication, and SEC retention.
- By the rule, no held-out evaluation and no controls were run. Held-out folds 1–3 remain untouched
  across all of Gate 3.

### What was built (from `edit_dev` only)

| block ℓ | read: AUROC on 10% slice | mean rᵀh, absent / answerable | rms rᵀh, generic text | write: log P(target), before → after | ‖δ‖ / ‖residual‖ |
|---|---|---|---|---|---|
| 5 | 0.897 | 0.69 / 0.31 | 0.138 | −24.9 → −0.29 | 1.08 |
| 6 | 0.919 | 0.71 / 0.29 | 0.138 | −24.9 → −0.25 | 1.21 |
| 7 | 0.916 | 0.71 / 0.28 | 0.136 | −24.9 → −0.74 | 1.45 |
| 8 | 0.904 | 0.71 / 0.29 | 0.136 | −24.9 → −1.15 | 1.54 |

- λ_rel = 1e-4 (the smallest in the grid) was chosen at every layer.
- The generic penalty used 300,953 prompt positions.

### Dev-fold grid (`iter_val_0`)

| config | safety | ‖ΔW‖/‖W‖ (that matrix) | result |
|---|---|---|---|
| **L5, γ 0.5** | pass | 0.226 | the evaluation below |
| L5, γ 1 / 2 | veto | 0.45 / 0.91 | — |
| L6, γ 0.5 / 1 / 2 | veto | 0.25 / 0.51 / 1.01 | — (γ 2: canary loss +10.8%) |
| L7, γ 0.5 / 1 / 2 | veto | 0.31 / 0.62 / 1.24 | — (γ 2: canary loss +12.4%) |
| L8, γ 0.5 / 1 / 2 | veto | 0.33 / 0.65 / 1.31 | — (γ 2: canary loss +11.5%) |

L5, γ 0.5 vs θ_qa, paired (95% CI):

| metric | Δ |
|---|---|
| abstention, absent 1 / 2 / 3 docs | +0.187 / +0.193 / +0.196 (all significant) |
| discrimination k = 1 / 2 / 3 | +0.184 / +0.184 / +0.172 → **valid** (≥ half the gain) |
| false abstention, answerable | +0.013 [+0.007, +0.020] |
| clean F1 | **−0.023 [−0.046, −0.001] → FAIL** (the lower bound must be ≥ −0.02) |
| clean fabrication | **+0.036 [+0.009, +0.063] → FAIL** |
| retention sec / case-law / web | **+0.63%** / +0.37% / +0.25% → **FAIL** (sec > +0.5%) |
| global ‖Δθ‖/‖θ‖ | 0.025 (a single matrix) |

### Diagnosis (post-hoc; not a basis for any claim)

1. **The read is linear, but the abstention decision needs a threshold.** The read fires at 0.69 on
   absent items and still at 0.29 on answerable ones. Every answerable item gets about 40% of the
   push towards abstaining.
   - Most answerable items stay below the decoding flip, so false abstention rises by only +0.013.
   - But their answers are perturbed: 203 of 332 clean answers changed text. That is where the
     clean-F1 and fabrication failures come from.
   - A rank-1 change to `W_down` has no nonlinearity of its own to cut the 0.29 to zero.
2. **The write had to be large.** The optimised δ is 1.1–1.5× the norm of the residual state it is
   added to. With an unbounded read norm (the smallest λ won every time), the outer product exceeds
   the per-tensor safety limit at any useful γ. The design never constrained the edit's norm. In
   hindsight, that was a design flaw.
3. **The forgetting per unit of abstention is much better than the LoRA's**, though still not good
   enough.
   - At about the same abstention (+0.19), the scaled LoRA would need s ≈ 0.45. The Gate 3a fit puts
     that at about +2.2% retention.
   - The rank-1 edit costs +0.42% on average, roughly **5× less forgetting** for the same gain.
   - This is dev-fold and exploratory, but it is the first edit to move the Gate 3a frontier at all.
4. **Copying behaviour:** the edited model often writes variants such as "not stated in the
   documents" or "...in the context of the context". The metric counts these as abstention, but the
   write vector does not pin the exact string.

### What this suggests (not run; it would need its own pre-registration)

**A thresholded single-neuron edit** (in the spirit of T-Patcher, Huang et al. 2023, which adds one
neuron per edit). Instead of a linear rank-1 change to `W_down`, repurpose one MLP neuron in block 5:
- its gate/up rows read the probe direction with a bias, so SiLU keeps it ≈ 0 unless the absence
  score clears a threshold;
- its `W_down` column writes a **norm-bounded** δ.

This addresses diagnosis 1 (a real threshold, so answerable items get ~0 push) and diagnosis 2 (δ is
bounded, and the change spans three single rows or columns rather than a full-norm outer product).
The cost of repurposing the neuron is measured by retention, like everything else.

# Gate 3b-iii — a thresholded single-neuron abstention edit

**Approved 2026-11-10** (after the rank-1 edit's negative result, docs/11). **Part A was written
before any edit is built or evaluated.** Its construction-feasibility facts come from `edit_dev` only.

## Part A — pre-registration

**Why this design.** It fixes the two diagnosed failures of the rank-1 edit:

1. **A linear read had no threshold**, so answerable items still received about 40% of the push.
   *Fix:* the read goes through the MLP's own SiLU gate, with a bias, so the neuron is silent unless the
   absence score clears a threshold.
2. **The write was unbounded** (‖δ‖ = 1.1–1.5× the residual norm), and the edit broke the per-tensor
   safety limit. *Fix:* δ is optimised under a hard norm bound, and the edit touches only three
   vectors.

**The edit.** Repurpose **one existing neuron j** of block ℓ's MLP. Each output of
`W_down · (SiLU(W_gate x̂) ⊙ (W_up x̂))` is a sum over neurons, so neuron j contributes
`W_down[:, j] · SiLU(g_jᵀx̂) · (u_jᵀx̂)`. Three vectors are replaced, and nothing else changes:

- **gate row g_j = s · (w − τ·c)**, with s = 4.
  - w is a logistic absence probe on x̂, the MLP input at block ℓ, fit **without an intercept**.
  - Positives: the last prompt token of `edit_dev` absent items.
  - Negatives: the last token of `edit_dev` answerable items, plus generic positions (every 8th
    non-final prompt position of 2,000 `edit_dev` items).
  - The two negative groups carry equal total weight, and the positives match them.
  - c is the near-constant direction (c·x̂ ≈ 1; ridge fit on the same positions; in the feasibility
    check, mean 1.00 and std 0.04). It lets the gate subtract a threshold τ.
  - L2 strength λ ∈ {1e-4, 1e-3, 1e-2}, chosen by absent-vs-answerable AUROC on the fixed 10% seed
    slice of `edit_dev` (the same slice as docs/11).
- **up row u_j = c**, so u_jᵀx̂ ≈ 1 and the neuron's activation is about SiLU(s·(score − τ)).
- **down column W_down[:, j] = δ / m**, where m is the median activation of the neuron over `edit_dev`
  absent items that fire.
  - δ is the norm-bounded write vector: added to block ℓ's output at the last prompt token, it
    maximises the teacher-forced log P("not stated in the context<|eos|>") on `edit_dev` absent items.
  - Optimisation: projected Adam, 300 steps, batch 16, seed 1337, with ‖δ‖ ≤ ρ · (rms residual norm
    at that layer).
- **Which neuron:** the one with the smallest mean |activation| × ‖W_down[:, j]‖ over the generic
  positions. That is the neuron whose loss should cost the least.

**Grid (dev fold `iter_val_0`, 12 configurations).** Block ℓ ∈ {5, 6} × bound ρ ∈ {0.25, 0.5, 1.0} ×
threshold τ ∈ {0, 1}, in probe-logit units.

**Pipeline, selection and stopping rule:** unchanged from docs/11.
- The safety gate, the approved guardrails and the validity condition.
- The largest abstention gain among configurations that pass.
- **If none passes, stop**: no held-out evaluation.

**Report for a selected edit.**
- Held-out folds `iter_val_1..3` pooled, paired vs θ_qa.
- 3 random controls, matched in rank and norm per tensor (via `random_like`). Unlike the edit, these
  spread across the whole matrix, not one neuron.
- **Structured controls**, on dev and held-out:
  - **(a) shuffled read:** the probe is refit on shuffled labels, with the generic negatives kept;
  - **(b) write-only:** the probe is fit with every last token positive, so the neuron fires at the
    start of *every* reply;
  - **(c) ablation only:** neuron j is zeroed and nothing is written. This is the cost of
    repurposing the neuron.
- **Milestone:** full retention on both disjoint held-out sets, plus the copy and induction probes.

**What counts as a positive result.** On the held-out folds, the selected edit must:
1. pass every approved guardrail, including full retention ≤ +0.5%;
2. pass the validity condition;
3. beat the 3 random controls and controls (a) and (b) on abstention.

**Pre-run amendment (2026-11-10, before any dev-fold evaluation).**
- **Change:** the gate sharpness s is no longer fixed at 4. For each layer it is the largest s ≤ 4
  with ‖Δ gate row‖ ≤ 0.2·‖W_gate‖ at τ = 1, found by bisection on the real probe. The controls
  reuse that s.
- **Why:** a smoke test on `edit_dev` items only (no dev or held-out data touched) showed that s = 4
  makes the new gate row **1.1×** the norm of the whole W_gate matrix. The per-tensor safety limit is
  0.25, so every configuration would have been vetoed, which is the rank-1 edit's failure mode.
- **The cost:** a softer threshold. Generic positions may leak a small negative SiLU activation. The
  build log reports this (`generic_mean_abs_silu`), and retention measures it.
- **The same smoke test, informally:** after only 40 optimisation steps, the edit fired on 99.6% of
  absent and 1.8% of answerable `edit_dev` items (in-sample). On unseen `edit_dev` items it abstained
  on 10% of absent and 5% of answerable items. That is a construction check, not a result.

**Known limit, as in docs/11.** On this item pool, the absence signal is roughly word overlap.

## Part B — results (`wsp_runs/gate3b_neuron/`: `build.log`, `run.log`, `results.json`, `vectors/build_meta.json`)

**Verdict under the pre-registered rule: NEGATIVE.**
- No configuration passed. By the rule, no held-out evaluation and no controls were run. Held-out
  folds 1–3 remain untouched.
- But this is **by far the closest result so far**. The best configuration passes every guardrail
  except one domain of retention.

### What was built (from `edit_dev` only)

| | block 5 | block 6 |
|---|---|---|
| neuron repurposed (least used on generic text) | 3042 | 1880 |
| gate sharpness s (cap: Δ row ≤ 0.2·‖W_gate‖) | 0.42 | 1.10 |
| probe AUROC, 10% slice | 0.925 | 0.930 |
| firing at τ = 0: absent / answerable / generic positions | 94% / 18% / 0.1% | 94% / 20% / 0.2% |
| write log P(target) at ρ = 0.25 / 0.5 / 1.0 | −14.8 / −2.8 / −0.32 | −15.5 / −5.4 / −0.35 |

### Dev-fold grid (`iter_val_0`)

| config | abstain Δ | discrimination min | clean F1 Δ | false-abstain Δ | retention sec / case-law / web | outcome |
|---|---|---|---|---|---|---|
| L5 ρ0.25 τ0 | +0.047 | +0.003 | +0.005 | +0.001 | +0.62 / +0.15 / +0.10% | fail: retention, validity |
| L5 ρ0.25 τ1 | +0.038 | +0.003 | +0.011 | +0.001 | +0.83 / +0.21 / +0.14% | fail: retention, validity |
| L5 ρ0.5 τ0 | +0.502 | +0.380 | −0.008 | +0.029 | +3.77 / +0.93 / +0.50% | fail: retention, clean F1 |
| L5 ρ0.5 τ1, ρ1 τ0, ρ1 τ1 | — | — | — | — | — | safety veto (down column) |
| L6 ρ0.25 τ0 | +0.008 | 0.000 | +0.002 | +0.001 | +0.02 / +0.01 / +0.01% | fail: validity (too weak) |
| L6 ρ0.25 τ1 | +0.008 | 0.000 | +0.012 | 0.000 | +0.03 / +0.01 / +0.01% | fail: validity (too weak) |
| L6 ρ0.5 τ0 | +0.521 | +0.431 | +0.005 | +0.027 | **+1.44** / +0.40 / +0.24% | fail: SEC retention only |
| L6 ρ0.5 τ1 | +0.477 | +0.416 | +0.008 | +0.017 | **+1.72** / +0.45 / +0.27% | fail: SEC retention only |
| **L6 ρ1 τ0** | **+0.726** | **+0.642** | +0.005 | +0.066 | **+1.72** / +0.48 / +0.32% | **fail: SEC retention only** |
| L6 ρ1 τ1 | — | — | — | — | — | safety veto (gate-row tensor 0.289) |

**Where it stands against every earlier edit** (dev fold; mean retention is over the three domains):

| edit | weights changed | abstain Δ | discrimination min | clean F1 Δ | mean retention |
|---|---|---|---|---|---|
| LoRA lr 3e-4 (Gate 2) | 84 tensors | +0.925 | +0.77 | +0.10 | +8.1% |
| scaled LoRA, s = 0.5 (Gate 3a) | 84 tensors | +0.303 | +0.27 | +0.09 | +2.5% |
| rank-1, L5 γ0.5 (Gate 3b-ii) | 1 matrix | +0.192 | +0.17 | −0.023 | +0.42% |
| **single neuron, L6 ρ1 τ0** | **3 vectors** | **+0.726** | **+0.64** | **+0.005** | **+0.84%** |

At comparable abstention, the neuron edit costs roughly a tenth of the scaled LoRA's forgetting. It
does not damage answering.

### Diagnosis (post-hoc, dev-fold retention windows only; not a basis for any claim)

On 64 windows per domain, for the L6 ρ1 τ0 edit:

| | sec | case-law | web |
|---|---|---|---|
| neuron fires (activation > 0.5) at this share of tokens | **0.92%** | 0.32% | 0.34% |
| loss Δ at the firing token / the next token | +0.99 / +0.42 | +0.95 / +0.27 | +0.74 / +0.27 |
| share of the domain's total loss increase at firing tokens | 33% | 28% | 24% |
| **zeroing neuron 1880 alone (no write)** | loss Δ **0.0000** | 0.0000 | 0.0001 |

- **Repurposing the neuron is free.** Its original function cost nothing measurable.
- **The cost comes from misfires.** The new neuron occasionally fires inside ordinary text: `,` and
  `of` in SEC filings, `.` in case law, `-` in web text. Each misfire writes the abstain push into
  the middle of a document, and later positions inherit the damage. SEC filings trigger it about 3×
  as often as the other domains.
- **The gate encodes the wrong thing.** It encodes "this context looks like information is missing",
  not "this is the start of a reply". Its generic negatives were prompt positions inside the QA
  template, never raw filing text.

### Next design (not run; would need its own pre-registration)

**Use the SwiGLU product as an AND gate.** The MLP computes `SiLU(gate·x̂) × (up·x̂)`.
- At present the up row is a constant, c.
- Instead, make it a **reply-start detector**: a ridge read that is ≈ 1 at the last prompt token (the
  `\n` after `<|assistant|>`) and ≈ 0 everywhere else, with raw document text from `edit_dev` as
  extra negatives.
- The neuron would then fire only where *both* "the answer is absent" *and* "a reply is starting"
  hold.
- The model's own multiplicative gating does this; no new mechanism is needed.
- The misfires at `,`, `of` and `.` would be multiplied by roughly zero.

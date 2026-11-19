# Gate 3a Report: deterministic weight-space search, a negative result

**Question (pre-registered in docs/08_GATE3_PLAN.md):** can a scaled or masked version of the Gate 2
LoRA delta keep a real share of its abstention gain while staying inside every approved guardrail?

**Answer: no.** All 18 configurations were scored on the dev fold: 10 scale settings, 6 single
groups and 2 efficient subsets. **None** reaches a valid abstention gain inside the +0.5% retention
budget.

- **Where the budget runs out, abstention is still zero.** The budget is reached at about 16% of the
  lr 3e-4 delta and 26% of the lr 1e-4 delta, where abstention is +0.004 or less.
- **No single layer or module group carries the skill.** Each of the six groups on its own gives at
  most +0.073 abstention, against +0.925 when they act together.
- **θ_qa does not hold the skill in reserve.** A teacher-forced post-hoc diagnostic finds no latent
  "answer absent" signal behind the target string: AUROC 0.42, below chance. The signal appears only
  as the edit grows.

By the pre-registered rule, Gate 3a stops here: no held-out evaluation and no random-control run. Nothing
was selected.

Tables: `wsp_runs/gate3a/GATE3A_TABLES.md`, from `python -m wsp.gate3a_report`. Raw data:
`results.json`, `gate3a_numbers.json`, `log.jsonl`, `diagnostic_abstain_logprob.json`. The originals'
lock was verified at the start of every script. No new training was run; every edit is θ_qa + a
scaled and/or masked copy of a saved Gate 2 delta.

---

## 1. Phase A: scale sweep, θ_qa + s·Δ (dev fold `iter_val_0`)

| s | lr 3e-4: abstain Δ | discrimination min | clean F1 Δ | retention (mean of 3 domains) | lr 1e-4: abstain Δ | retention |
|---|---|---|---|---|---|---|
| 0.05 | +0.001 | 0.000 | +0.020 | +0.11% | +0.000 | +0.05% |
| 0.1 | +0.002 | 0.000 | +0.035 | +0.28% | +0.001 | +0.14% |
| 0.2 | +0.004 | 0.000 | +0.063 | **+0.67%** | +0.002 | +0.37% |
| 0.3 | +0.013 | +0.012 | +0.072 | +1.18% | +0.003 | **+0.63%** |
| 0.5 | +0.303 | +0.271 | +0.089 | +2.51% | +0.057 | +1.28% |
| 1.0 (Gate 2) | +0.925 | +0.768 | +0.096 | +8.14% | +0.856 | +3.75% |

Bold marks the first scale that breaks the +0.5% budget. A configuration passes only if *every*
domain is ≤ +0.5%, so the per-domain table in `GATE3A_TABLES.md` is the authority.

- **The two curves have different shapes.** Abstention is a **threshold** response: almost nothing up
  to s = 0.3, then a steep rise. The retention cost grows smoothly with scale.
  - The cost fits a·s + b·s² well (lr 3e-4: 2.06·s + 6.08·s² %).
  - That puts the +0.5% budget at **s ≈ 0.16** for lr 3e-4 and **s ≈ 0.26** for lr 1e-4.
  - At those scales abstention is at most +0.004, and discrimination is 0.
- **Clean F1 behaves the other way.** It rises early and smoothly, and is largely spent by s ≈ 0.2:
  +0.063 at a retention cost of +0.37% (lr 1e-4). Inside the budget, the delta mostly delivers the
  answer-quality and style part of the change, not abstention.

## 2. Phase B: attribution, Δ_lr3e-4 restricted to one group at scale 1.0

| group | ‖Δθ‖/‖θ‖ | abstain Δ | discrimination min | clean F1 Δ | false-abstain Δ | retention sec / case-law / web |
|---|---|---|---|---|---|---|
| attn L0–3 | 0.0085 | +0.000 | 0.000 | +0.065 | 0.000 | +1.34 / +0.77 / +0.67% |
| attn L4–7 | 0.0094 | +0.000 | −0.003 | +0.067 | +0.001 | +1.11 / +1.01 / +1.12% |
| attn L8–11 | 0.0097 | +0.001 | 0.000 | +0.032 | 0.000 | +0.23 / +0.31 / +0.26% |
| mlp L0–3 | 0.0142 | −0.001 | 0.000 | +0.071 | 0.000 | +1.70 / +1.18 / +1.12% |
| mlp L4–7 | 0.0153 | +0.000 | 0.000 | +0.067 | 0.000 | +0.77 / +0.84 / +1.06% |
| mlp L8–11 | 0.0162 | **+0.073** | +0.021 | +0.057 | +0.037 | +0.52 / +0.77 / +0.85% |

- **The skill is not localised; it is synergistic.** The six groups partition the delta exactly (all
  84 tensors). Their solo abstention gains sum to **+0.073**, against **+0.925** for the whole delta.
  Abstention needs the groups *together*, so it cannot be bought cheaply by editing one place.
- **The only group with any effect is not real abstention.** The late MLPs (L8–11) raise abstention
  by +0.073, but discrimination is only +0.021 and false abstention rises by +0.037. That group mostly
  raises the abstain response everywhere. It fails the pre-registered validity condition
  (discrimination ≥ half the gain).
- **Early layers buy the most forgetting.** They carry the largest retention cost per unit of norm,
  for no abstention: attn L0–3 costs +1.34% on SEC, mlp L0–3 +1.70%.

## 3. Phase C: efficient subsets

Only one group (mlp L8–11) had a significant abstention gain, so, per the pre-registered rule, only
the top-1 union exists:

- at s = 0.25: abstention +0.000, retention ≤ +0.12%;
- at s = 0.5: abstention +0.002, retention ≤ +0.30%.

Both pass the guardrails. Neither does anything for the target.

## 4. Post-hoc diagnostic (exploratory, not pre-registered, dev fold only)

**Purpose.** Was abstention absent at small scale only because greedy decoding had not yet *flipped*?
**Method.** For each item, I scored log P("not stated in the context<|eos|>" | prompt) by teacher
forcing. AUROC measures how well that score separates absent from answerable items at the same
document count (0.5 = no signal).

| scale of Δ_lr3e-4 | AUROC k=1 / 2 / 3 | pooled | mean log P, absent / answerable |
|---|---|---|---|
| 0 (θ_qa) | 0.354 / 0.397 / 0.445 | **0.422** | −24.95 / −24.03 |
| 0.1 | 0.474 / 0.479 / 0.491 | 0.485 | −14.26 / −14.13 |
| 0.15 (≈ budget) | 0.550 / 0.540 / 0.523 | **0.534** | −10.50 / −10.75 |
| 0.3 | 0.744 / 0.720 / 0.682 | 0.715 | −5.95 / −7.29 |
| 0.5 | 0.937 / 0.921 / 0.895 | 0.918 | −2.70 / −6.09 |
| 1.0 | 0.983 / 0.976 / 0.969 | 0.976 | −0.26 / −4.68 |

- **θ_qa has no latent abstention preference**; it leans the wrong way, at 0.42. It finds the abstain
  string slightly *more* likely when the answer is present.
- **What a small edit does is general, not targeted.** It raises the string's likelihood everywhere:
  −25 → −10.5 nats at s = 0.15, with almost no separation (AUROC 0.53).
- **Separation grows with the size of the edit.** The greedy threshold is not hiding a signal that
  is already there.
- **What this does not show.** It tests the abstain *string*. It does not test whether θ_qa's hidden
  states represent answer-absence. A linear probe on activations would test that (§6).

## 5. What Gate 3a establishes

- **Within the LoRA's own direction, the target is out of reach.** Scaling and per-layer/module
  masking (the knobs an editor LLM would control over this delta) cannot reach valid abstention under
  the approved retention budget. An LLM controller searching the same knobs cannot do better than
  exhaustively evaluating them, and this lattice has no feasible point. **So editor-LLM arms over these
  knobs are not worth running.**
- **This is a property of this delta, not of every possible edit.** The LoRA was trained with no
  retention constraint. A direction found *with* one might be cheaper, which is untested. The negative
  result is about the LoRA's subspace, scales and masks.
- **Abstention is a new skill for θ_qa, not an unlocked one.** θ_qa was never trained to abstain
  (0 of 15,000 SFT targets), and the string-level diagnostic shows no latent preference to sharpen.
  Gate 3a is consistent with the plan's earlier expectation that "install" targets are harder than
  "sharpen" targets.
- **A side observation, not a claim.** Inside the budget, the delta raises dev-fold clean F1. The
  largest such gain is +0.063 at s = 0.2 for lr 1e-4, at a retention cost of +0.33–0.40%.
  - That is a cheap answer-quality effect, not the approved target.
  - It carries the same caveat as Gate 2: RAFT-style references, measured on the dev fold only.
  - Claiming it would need its own pre-registration: a held-out fold, random controls and the frozen
    eval.

## 6. Options for Gate 3b (decision needed)

1. **Linear probe of θ_qa's activations for answer-absence** (recommended first: about 30 minutes,
   local, no edit).
   - If θ_qa internally represents "the answer is not in these documents" (high AUROC from a probe on
     hidden states), then a **representation-level edit** becomes possible. One example: a rank-1
     edit that reads that direction and writes the abstain response. That is a true, small, surgical
     weight-space edit, and the setting where an editor LLM could add value.
   - If θ_qa does not represent it, small edits are ruled out for this target at 125M. That is a
     clean negative result for the brief.
2. **Retention-constrained LoRA**, trained with the abstention loss plus a KL or replay term on
   held-out-domain text. This tests whether a *cheaper direction* exists at all. It is a gradient
   method, but a necessary baseline before any claim that weight-space programming beats gradients.
3. **Switch the target to a "sharpen" task such as clean answer quality**, where Gate 3a shows small
   in-budget edits do move the metric. This changes the approved primary target, so it needs your
   approval and a fresh pre-registration.

Not recommended now: editor-LLM arms over scale and mask knobs (§5).

## 🛑 GATE 3a ENDS HERE

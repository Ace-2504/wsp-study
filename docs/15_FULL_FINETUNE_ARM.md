# Full fine-tuning comparison arm

**Approved 2026-11-11** ("go with the original plan"). This is the full-SFT comparison named in the
plan (docs/03, docs/05). **Part A was written before any training or evaluation of this arm.**

## Part A — pre-registration

**Question.** The project compares weight editing against fine-tuning. Gate 2 used LoRA. This arm
trains *every* parameter of θ_qa on the same data, so the comparison also covers ordinary full
fine-tuning.

**Recipe** (identical to Gate 2 LoRA except that all parameters train and the learning rates differ):
- Start from θ_qa. Train on `edit_dev` v2 (12,732 items, the same shuffled order, seed 1337), 1 epoch.
- Loss on assistant tokens only (θ_qa's own renderer `ft_data.render_custom`).
- Micro-batch 4 × accumulation 4 (effective batch 16, as Gate 2). *Changed from 8 × 2 before any
  result was seen:* at 8 × 2 the full optimiser state filled the 12 GB GPU and training ran about 3×
  slower; the first run was stopped at step ~100 and discarded. AdamW (0.9, 0.95), weight decay 0. Cosine schedule with 3% warmup.
  Gradient clip 1.0. bf16 autocast over fp32 master weights.
- **Two learning rates: {1e-5, 3e-5}.** 3e-5 is the rate θ_qa itself was fine-tuned with
  (`vendor/finetuning/ft_config.py`); 1e-5 is the gentler option.
- The delta is the trained weights minus θ_qa's weights, over every tensor (the tied `lm_head` is
  the embedding).

**Selection** (Gate 2's rule, on `iter_val_0` only):
- Among the configs that pass the safety gate and every approved guardrail, pick the one with the
  largest abstention gain.
- If none passes, continue with the one with the largest abstention gain, for the record, flagged
  as failing.

**Controls.** Three norm-matched random deltas of the selected arm (`engine.random_like`, full rank,
same per-tensor norm), scored on `iter_val_0` and the held-out folds.

**Report** (nothing selected on these):
- Held-out `iter_val_1..3` pooled, paired against θ_qa.
- Full retention (both disjoint sets) and the verbatim-copy probe for the selected arm and random
  seed 0.
- The overlap-matched pool (docs/14) for the selected arm, with the docs/14 verdict rule.

**Guardrails** are the approved ones: retention ≤ +0.5% per domain; clean-F1 paired Δ lower CI ≥
−0.02; verbatim copy ≥ 0.93; fabrication must not rise significantly.

**Prediction, stated in advance.** Full fine-tuning installs abstention at least as strongly as LoRA,
and fails retention by more than LoRA did, because it changes every tensor. On the matched pool it
is PARTIAL, like LoRA.

## Part B — results (`wsp_runs/gate4_fullft/`: `log.jsonl`, `reports/`, `retention/`, `probes/`, `gate4_fullft_numbers.json`; matched pool in `wsp_runs/ovm_pool/ovm_numbers.json`)

**Verdict: full fine-tuning fails the approved retention guardrail, like LoRA.** It installs the
strongest abstention of any arm, and it improves answering, but held-out loss rises by +4.9% to +5.9%
against a +0.5% budget. On the overlap-matched pool it is **PARTIAL**, almost identical to LoRA.

### Training and selection (`iter_val_0`)

| config | ‖Δθ‖/‖θ‖ | final train loss | wall | iter_val_0 |
|---|---|---|---|---|
| lr 1e-5 | 0.0076 | 0.470 | 14 min | abstain Δ +0.936 [+0.921, +0.951]; clean F1 Δ +0.104; **fails retention** (fast subset) |
| lr 3e-5 | 0.0201 | 0.283 | 14 min | **safety veto: canary loss +15.6%** (limit +10%); not evaluated |

- **Neither config passed.** Under the pre-registered fallback, lr 1e-5 was carried forward for the
  record and is **flagged as failing**.
- **The controls are at the selected arm's norm (0.0076).** All three random deltas leave abstention
  at 0.000 on `iter_val_0`.

### Held-out `iter_val_1..3` pooled, paired against θ_qa (3,378 absent and 3,378 answerable items)

| | full FT lr 1e-5 | random s0 / s1 / s2 |
|---|---|---|
| abstention Δ on absent items | **+0.960 [+0.953, +0.966]** | 0.000 for all three seeds on every fold |
| abstain rate abs1 / abs2 / abs3 | 0.971 / 0.957 / 0.951 | 0.000 |
| false abstention, ans1 / ans2 / ans3 | 0.100 / 0.103 / 0.135 (Δ +0.112) | ≈ 0 |
| discrimination k = 1 / 2 / 3 | +0.870 / +0.854 / +0.816 | ≈ 0 |
| clean F1 Δ | **+0.105 [+0.086, +0.124]** | −0.010 to +0.008 per fold |
| clean fabrication Δ | −0.053 [−0.069, −0.037] | — |

### Retention (full sets, both disjoint corpora) and copy

| arm | e4 sec | e4 case-law | e4 web | v1 case-law | v1 web | verbatim copy |
|---|---|---|---|---|---|---|
| full FT lr 1e-5 | **+5.18%** | **+4.89%** | **+5.82%** | **+4.94%** | **+5.86%** | 0.960 |
| random s0, same norm | +0.03% | +0.02% | +0.02% | +0.02% | +0.02% | 0.965 |

- **The cost comes from the direction of the change, not from its size.** A random change of the
  same norm costs 0.02–0.03%.
- **Per unit of change, full FT forgets more than LoRA.** LoRA lr 1e-4 moved the weights by 0.0113
  and cost +3.3% to +4.2% (docs/07). Full FT moved them by only 0.0076 and cost +4.9% to +5.9%.

### Overlap-matched pool (docs/14 verdict rule)

| model | D on pool v2 | **D matched** | survives | D deleted-only | abstain del1 / del2 | abstain ans1 / ans2 | verdict |
|---|---|---|---|---|---|---|---|
| AND-gate edit | +0.720 | +0.122 [+0.097, +0.150] | 17% | +0.061 | 0.088 / 0.118 | 0.034 / 0.051 | OVERLAP-DOMINATED |
| LoRA lr 3e-4 | +0.882 | +0.263 [+0.224, +0.300] | 30% | +0.178 | 0.271 / 0.295 | 0.107 / 0.104 | PARTIAL |
| **full FT lr 1e-5** | +0.862 | **+0.254 [+0.216, +0.291]** | **29%** | +0.170 | 0.293 / 0.312 | 0.139 / 0.127 | **PARTIAL** |

Matched-answerable F1 Δ vs θ_qa: full FT +0.080 [+0.062, +0.098]; LoRA +0.084; AND-gate −0.013.

### Comparison of every arm (held-out, pooled)

| arm | weights changed | abstain Δ | clean F1 Δ | worst retention | matched-pool D | passes all guardrails |
|---|---|---|---|---|---|---|
| LoRA lr 1e-4 (Gate 2) | 7 projections × 12 layers, rank 16 | +0.856 | +0.102 | +4.17% | — | no (retention) |
| LoRA lr 3e-4 (Gate 2) | same | +0.925 | +0.096 | +8.87% | +0.263 | no (retention) |
| **full FT lr 1e-5** | every tensor | **+0.960** | **+0.105** | +5.86% | +0.254 | no (retention) |
| full FT lr 3e-5 | every tensor | — | — | — | — | safety veto |
| **AND-gate edit (docs/13)** | **3 vectors, 1 neuron** | +0.737 | −0.005 | **+0.011%** | +0.122 | **yes** |

### Prediction check

- **Correct:**
  - abstention at least as strong as LoRA (+0.960 vs +0.925);
  - fails retention;
  - PARTIAL on the matched pool.
- **Partly wrong:** full FT forgot *less* than LoRA lr 3e-4 (+5.9% vs +8.9%), because its change was
  smaller. Per unit of change it forgot more.

### What this means

1. **Gradient training, LoRA or full, buys stronger and somewhat more genuine abstention.** It also
   improves answering. The cost is 3–9% forgetting, 7–18× the budget, at every setting tried.
2. **Full FT and LoRA are indistinguishable on the question that matters.** About 30% of their
   discrimination survives when overlap carries no information.
3. **The AND-gate edit is the only arm inside every guardrail.** It forgets about 500× less than full
   FT, but it buys mostly overlap-based abstention and no answering gain.
4. **Nobody has solved the problem yet.** No arm gets abstention that goes beyond overlap while
   staying inside the retention budget.

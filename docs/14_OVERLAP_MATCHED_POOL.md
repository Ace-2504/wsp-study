# Overlap-matched control pool: does the installed abstention read answer-absence, or word overlap?

**Approved 2026-11-11** (the next step named in docs/13). **Part A was written before the held-out
pool is built and before any model is run on it.** The construction-feasibility numbers below come
from `edit_dev` seeds only (`wsp_runs/ovm_pool/feasibility_edit_dev.json`).

## Part A — pre-registration

**The problem.** On pool v2 the absent items' documents simply share fewer of the question's words.
A 6-feature lexical-overlap model, using no neural network, separates absent from answerable items at
AUROC **0.966** (docs/10). θ_qa's own layer-5 probe reaches 0.95. So every abstention result so far
(LoRA, the AND-gate edit) could be "refuse when the documents don't mention the question's words".
Pool v2 cannot tell that apart from "refuse when the answer is not there".

**The fix: absent items that look answerable.** For each held-out seed, built from the seed's own
pool-v2 documents (no new text):

| kind | documents | label |
|---|---|---|
| ans1 / ans2 | golden / golden + D1 (pool v2) | answerable |
| abs1 / abs2 | D1 / D1 + D2 (pool v2, topical distractors) | absent |
| **del1 / del2** | **golden with the answer sentence deleted** / the same, + D1 (ans2's order) | **absent** |

- **The deletion.** Remove every sentence that overlaps the quoted support span. The passage keeps its
  topic and most of the question's words, but loses the answer.
- **The deletion is rejected (the seed is dropped) if any of these is true:**
  - fewer than 30 content words remain;
  - the quote or the normalised short answer still appears;
  - every content word of the answer still appears;
  - any number in the answer still appears.

**Matching.**
- **The score:** the docs/10 lexical model, refit identically (pool-v2 `edit_dev` features,
  standardised, λ = 1e-4). This is the strongest overlap rule we have.
- **The bins:** for each document count k ∈ {1, 2}, 20 quantile bins of its logit.
- **The draw:** in each bin, min(#absent, #answerable) items of each label, seed 1337.
- **The result:** absent and answerable items with the same overlap-score distribution. Inside this
  set the overlap rule is at chance by construction.

**Feasibility (from `edit_dev` seeds only, not the evaluation data).**
- 1,802 of 2,122 seeds survive deletion. The drops: 103 all answer words remain, 88 too short, 66
  answer string remains, 63 answer number remains.
- The matched set has 4,692 items: 2,346 answerable, 1,720 deleted-answer and 626 distractor absent.
- Lexical AUROC on the matched set: **0.502 [0.485, 0.520]**. Before matching it is 0.966 (pool v2).
  Answerable vs deleted alone, unmatched: 0.762.

**What is run** (on the held-out seeds of `iter_val_1..3`; nothing is selected on this pool):
- **Models**, using the generation and scoring protocol of every gate (batched greedy,
  `metrics.score_item`):
  - θ_qa;
  - the **AND-gate edit** (`gate3b_andgate/deltas/selected.safetensors`);
  - the **Gate 2 LoRA lr3e-4** (`gate2/deltas/lora_lr0.0003.safetensors`) as the fine-tuning
    reference.
- **Representation checks** (no generation): AUROC on the matched set for
  - θ_qa's layer-5 last-token probe, the docs/10 configuration (λ = 0.01, fit on pool-v2
    `edit_dev`);
  - the AND-gate edit's own read, the gate pre-activation S·wᵀx̂ at block 6.
- **Label audit by an independent agent.**
  - The sample: 60 matched del1 items and 30 matched ans1 items, shuffled, with ids and kinds hidden.
  - The agent is a fresh one that never saw this project, and it answers per item: does the passage
    contain the answer to the question (yes/no, with the supporting sentence if yes)?
  - *Contamination* = its "yes" rate on del1. *Sensitivity* = its "yes" rate on ans1.
  - If contamination > 20%, the deleted-answer labels are too noisy for a firm verdict, and the
    result is reported as indicative only.

**Metrics.**
- **Pool check (must hold first):** lexical AUROC on the matched held-out set, with a seed-cluster
  bootstrap 95% CI. The pool is valid only if the CI contains 0.5, or the point is within 0.45–0.55.
  If not, the pool fails and no behavioural verdict is drawn.
- **Primary, for each model:** matched discrimination D_m = abstain(matched absent) −
  abstain(matched answerable), with a seed-cluster bootstrap CI.
- **Reference:** the same model's discrimination on the unmatched held-out pool-v2 items for
  k ∈ {1, 2} (abs1–2 vs ans1–2), D_v2.
- **Secondary:**
  - abstention on matched del vs matched distractor items;
  - false abstention and paired F1 on matched answerable items vs θ_qa;
  - representation AUROCs.

**Verdict rule** (applied to the AND-gate edit, and separately to the LoRA):
- **BEYOND OVERLAP:** lower CI of D_m ≥ 0.5 · D_v2. At least half the discrimination survives when
  overlap carries no information.
- **OVERLAP-DOMINATED:** upper CI of D_m < 0.25 · D_v2.
- **PARTIAL:** anything else.

**Prediction, stated in advance.** OVERLAP-DOMINATED for the AND-gate edit. Its read is a linear
probe at block 6, and docs/10 found the probe no better than overlap. LoRA is uncertain. This is a
test that can go against the edit, and it is reported either way.

**What this does not test.** Generalisation to other kinds of absence (a wrong entity, a wrong date,
a contradiction). Deletion is one way of making absent items look answerable, not every way.

## Part B — results (`wsp_runs/ovm_pool/`: `build.json`, `run.log`, `gen_*.json`, `probe.json`, `audit_*`, `ovm_numbers.json`)

**Verdict under the pre-registered rule.**
- **AND-gate edit: OVERLAP-DOMINATED**, as predicted.
- **Gate 2 LoRA: PARTIAL.**
- Most of the abstention installed so far is the word-overlap rule. A small part, real but weak, goes
  beyond it.

### The pool passed its checks

- **Built** from the held-out seeds of `iter_val_1..3`: 942 of 1,126 seeds survive the deletion
  filters. The drops: 55 too short, 44 answer string remains, 44 all answer words remain, 41 answer
  number remains.
- **Matched set:** 2,238 items.
  - Answerable: 1,119 (553 ans1, 566 ans2).
  - Absent: 1,119, of which **938 are deleted-answer** (454 del1, 484 del2) and 181 are topical
    distractors (99 abs1, 82 abs2).
- **Pool check:** lexical AUROC on the matched set is **0.503 [0.478, 0.531]**, down from **0.965
  [0.957, 0.973]** on the same seeds' pool-v2 items. The pool is **valid**: the overlap rule is at
  chance.
- **Blind label audit** by a fresh agent, unblinded only afterwards:
  - **Sensitivity:** it found the answer in **30 of 30** real answerable passages.
  - **Contamination:** it judged **7 of 60 (11.7%)** deleted-answer passages still answerable, under
    the 20% limit, so the labels support a firm verdict. Several of the 7 are inferences (e.g. a
    yes/no answer implied by a remaining sentence) rather than the deleted fact.

### Behaviour on the matched pool vs pool v2 (same models, same held-out seeds)

| model | D on pool v2 (k = 1, 2) | **D on the matched pool** | matched / v2 | D, deleted-answer only | D, distractor only | verdict |
|---|---|---|---|---|---|---|
| θ_qa | −0.000 | −0.001 | — | −0.001 | −0.001 | n/a (never abstains) |
| **AND-gate edit** | +0.720 [+0.698, +0.742] | **+0.122 [+0.097, +0.150]** | **17%** | +0.061 [+0.037, +0.086] | +0.443 [+0.363, +0.524] | **OVERLAP-DOMINATED** (upper CI 0.150 < 0.25 × 0.720 = 0.180) |
| **LoRA lr3e-4** | +0.882 [+0.864, +0.900] | **+0.263 [+0.224, +0.300]** | **30%** | +0.178 [+0.139, +0.217] | +0.701 [+0.625, +0.778] | **PARTIAL** (lower CI 0.224 < 0.441; upper CI 0.300 ≥ 0.221) |

Abstention rate per kind, on the matched items:

| model | ans1 / ans2 (should be 0) | abs1 / abs2 (distractor) | **del1 / del2 (answer deleted)** |
|---|---|---|---|
| θ_qa | 0.002 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| AND-gate edit | 0.034 / 0.051 | 0.515 / 0.451 | **0.088 / 0.118** |
| LoRA lr3e-4 | 0.107 / 0.104 | 0.818 / 0.793 | **0.271 / 0.295** |

- **When the answer sentence is deleted but the topic stays, the AND-gate edit almost never
  abstains:** 9–12%, against 3–5% on answerable items. On pool v2 it abstained on 74% of absent
  items.
- **Its remaining matched discrimination comes mostly from the 181 distractor items.** These are
  distractors that happen to share many of the question's words. Even here the edit catches only
  about half (0.45–0.52), against 0.71–0.76 on pool v2.
- **LoRA does better** on deleted-answer items (27–30%), but most of its ability also disappears
  (30% survives).
- **The audit does not change this.** Treating all 11.7% of contaminated deletions as answerable
  would raise the AND-gate's deleted-only D by at most about a tenth (≈ 0.061 → 0.07).

### Representation: what the model knows, as against what the edit uses

| read-out on the matched pool (seed-cluster CI) | AUROC | on pool v2 |
|---|---|---|
| lexical-overlap model | 0.503 [0.478, 0.531] | 0.965 |
| θ_qa layer-5 last-token probe (docs/10 config) | **0.639 [0.617, 0.664]** | 0.95 |
| the AND-gate edit's own read, S·wᵀx̂ at block 6 | **0.643 [0.620, 0.667]** | (probe slice 0.93) |
| the AND-gate neuron's full activation | 0.609 [0.583, 0.634] | — |

The neuron fires (activation > 0.5) at these rates on the matched items:

| ans1 | ans2 | abs1 | abs2 | del1 | del2 |
|---|---|---|---|---|---|
| 14% | 17% | 71% | 72% | 23% | 29% |

- **θ_qa does carry some answer-absence information beyond word overlap** (0.64, clearly above
  chance), but far less than pool v2 suggested (0.95).
- **The edit reads exactly that signal** (0.64), so it inherits both its strength and its weakness.
  Most of what the probe "knew" on pool v2 was overlap.
- **The threshold then turns a weak signal into rare abstention.** The neuron fires on only 23–29%
  of deleted-answer items, and the write converts fewer still into the exact abstain response.

### Secondary: answering on the matched answerable items

These are the answerable items whose overlap looks most "absent-like"; matching selects them.
- **θ_qa:** F1 0.214.
- **AND-gate edit:** −0.013 [−0.021, −0.006]. The lower bound sits just past the −0.02 line on this
  hard subset. This is not the approved guardrail test, which is on clean ans1 items and passed in
  docs/13, but it is a real cost on hard items.
- **LoRA:** +0.084 [+0.067, +0.102], better than θ_qa here too.

### What this means for the project

1. **The docs/13 result stands as measured, and it now has a precise meaning.** The edit installs
   "abstain when the documents don't contain the question's words", together with a weak
   beyond-overlap component (D ≈ 0.06 on deleted-answer items). It forgets almost nothing while doing
   so. It does **not** install general answer-absence detection.
2. **This is not a defect of weight editing as such.** The edit can only route a signal the model
   already computes, and θ_qa's own beyond-overlap signal is weak (AUROC 0.64). Fine-tuning (LoRA) can
   build some new signal (30% survives), at about 700× the forgetting.
3. **Pool v2 overstated every abstention result in this project**, LoRA's included. Any future arm
   should be scored on the matched pool as well.
4. **A better edit target exists in principle.** A read trained to separate deleted-answer from
   answerable passages (built from `edit_dev` deletions, never from the held-out pool) could pick up
   whatever beyond-overlap signal θ_qa has. The ceiling is set by that 0.64 AUROC, unless a
   different layer or position carries more. That would need its own pre-registration.

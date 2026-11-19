# Gate 2 Report — the edit loop, the controls, and the first baseline (no editor LLM yet)

**Question:** before any LLM proposes an edit, does the edit loop measure the right things? And what
does a standard gradient method (LoRA) achieve on the approved target, *abstain when the answer is not
in the documents*, within the approved guardrails?

**Answer, in three parts:**

1. **The loop instrument passes every self-check** (§1).
2. **The first Gate 2 run was a shortcut, and the shortcut is also in the frozen eval and in
   `raft.jsonl`** (§2). The LoRA learned "2 documents → abstain", because in both datasets the
   answer-absent condition always has exactly 2 documents. The item pools were rebuilt with document
   count decorrelated. That run is preserved as Gate 2a.
3. **On the corrected pools, LoRA learns real abstention, but it fails the approved retention
   guardrail by 7 to 18×** (§3–§5).
   - **What it gains, held-out:** abstention rises from 0.00 to 0.94–0.96. It discriminates at every
     document count (+0.83 to +0.90). It is not a random-direction effect: same-size random edits
     score exactly 0.000.
   - **What it costs:** held-out loss rises by +3.3% to +8.9% against a +0.5% budget. That is more
     forgetting than the whole SFT run cost θ_e4 (+2.7% to +3.2%). **No configuration passes**, so
     Gate 2 ends with no approved arm.

Every number below is computed by `python -m wsp.gate2_report` from saved artifacts in
`wsp_runs/gate2/` (tables: `GATE2_TABLES.md`, raw: `gate2_numbers.json`, append-only log:
`log.jsonl`). Originals lock-verified at the start of every script. Hardware: one RTX 3060 12 GB.

---

## 0. What was pre-registered (before any arm ran)

- **Target:** abstention rate on answer-absent items (`abs1`/`abs2`/`abs3`: 1/2/3 documents, none
  containing the answer). Abstention as the primary target and the guardrails were approved at the
  Gate 1 green flag (docs/06 §9).
- **Guardrails (approved):**
  - held-out loss ≤ +0.5% in every domain, vs θ_qa;
  - paired clean-F1 Δ lower CI ≥ −0.02 (clean = `ans1`, one golden document);
  - verbatim-copy rate ≥ 0.93;
  - clean fabrication must not rise significantly.
- **Arms** (`src/wsp/arms_gate2.py` docstring):
  - **S2 LoRA:** r=16, α=32, all 7 linear projections, 1 epoch on `edit_dev`, two learning rates.
  - **S1 Random:** 3 seeds, the same tensors, rank and per-tensor norm as the selected LoRA; only the
    direction differs.
- **Selection:** on `iter_val_0` only.
- **Report:** `iter_val_1..3` pooled, never used for any choice, paired per item vs θ_qa.
- **Data:** item pools built programmatically from RAFT rows (no paid generation).
  - 3,580 seeds and 21,480 items, with 0 answer leaks and no eval-document reuse
    (`splits/item_pool_manifest.json` records the rules and the SHA).
  - Splits: `edit_dev` is 60% of seeds; `iter_val_0..3` share the other 40%.
  - Held-out folds 1–3 have 1,126 items per kind.

## 1. Instrument self-checks — **ALL PASS** (`wsp_runs/gate2/gate2_checks.json`)

| check | result |
|---|---|
| Zero-delta identity: `materialize(θ_qa, 0)` vs the locked checkpoint | max loss difference **0.0**; **100/100** generations byte-identical |
| Safety gate, zero delta | pass |
| Safety gate, NaN delta | **vetoed** (non-finite) |
| Safety gate, Δ = 10·W on every MLP matrix | **vetoed**: tensor ratio 10.0, global 6.8, activation ratio 1,552, canary loss +210% |
| Fast retention subset (512 windows per domain) vs full set, on the θ_e4→θ_qa edit | error 0.04 / 0.08 / 0.0003 percentage points (sec / case-law / web) |
| θ_qa baseline, all 4 folds | abstention 0.000–0.003 on every absent kind; clean F1 0.22–0.27 |

The engine bug found while building it (a zero delta was not identical, because RoPE buffers were
computed in a different precision) was fixed before this run. The check above is the proof.

## 2. Gate 2a: the first run learned a document-count shortcut (superseded, preserved)

The first run used pools that mirrored the frozen eval's layout. Answerable items had 1 or 3
documents; answer-absent items always had **2**. Both LoRAs reached ~100% held-out abstention. A
filled 2×3 probe (`src/wsp/shortcut_probe.py`) showed what they had actually learned:

| arm | abstain, answerable: 1 / 2 / 3 docs | abstain, absent: 1 / 2 / 3 docs |
|---|---|---|
| θ_qa | 0.001 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| LoRA lr 1e-4 | 0.012 / **0.992** / 0.027 | **0.010** / 0.997 / **0.012** |
| LoRA lr 3e-4 | 0.001 / **0.992** / 0.012 | **0.002** / 0.998 / **0.004** |
| random s0 | 0.002 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |

- **What the LoRA keys on is the document count, not the content.** It refuses 99% of *answerable*
  questions given 2 documents. It *answers* 99% of unanswerable ones given 1 or 3 documents.
- **The same confound is in the instruments this project inherited:**
  - In the frozen eval (`eval_messages.jsonl`), every `retrieval_failure` item has 2 documents. No
    other condition has 2.
  - In `raft.jsonl`, every answer-absent training row has 2 documents.
- **Consequence:** a model trained on RAFT-style data can score on the eval's abstention condition
  without learning abstention. The published `slm-125m-raft` abstention number cannot be read as an
  abstention skill until it is re-measured on a decorrelated set. Its `retrieval_failure` score is
  also document-contaminated (docs/06 addendum: RAFT reuses every eval `retrieval_failure` document).
- **The fix (pool v2):** every seed yields six items: answerable and absent at 1, 2 and 3 documents.
  Document count then carries no information. `discrimination(k) = abstain(abs_k) − abstain(ans_k)` is
  now a mandatory output of every evaluation. Real abstention is high at every k; a shortcut collapses
  at some k.
- **Gate 2a's run is preserved** in `wsp_runs/gate2a_doccount_confounded/`, together with the old
  pool (`data/derived/item_pool_v1_doccount_confounded.jsonl`) and its manifest. The shortcut finding
  can be reproduced from disk.

## 3. Dev fold (`iter_val_0`, selection only)

| arm | ‖Δθ‖/‖θ‖ | abstain Δ | discrimination k=1/2/3 | clean F1 Δ | false-abstain Δ (answerable) | retention sec / case-law / web | guardrails |
|---|---|---|---|---|---|---|---|
| LoRA lr 1e-4 | 0.0113 | +0.856 [+0.834, +0.878] | +0.82 / +0.78 / +0.75 | +0.102 [+0.071, +0.133] | +0.072 | **+4.17% / +3.26% / +3.82%** | FAIL: retention |
| LoRA lr 3e-4 | 0.0309 | +0.925 [+0.908, +0.941] | +0.84 / +0.83 / +0.77 | +0.096 [+0.066, +0.127] | +0.113 | **+8.87% / +7.55% / +7.99%** | FAIL: retention |
| random s0 (lr 3e-4 norm) | 0.0309 | −0.001 | 0 / 0 / 0 | **−0.044 [−0.068, −0.022]** | 0.000 | +0.60% / +0.37% / +0.25% | FAIL: clean F1, retention |
| random s1 | 0.0309 | +0.002 | 0 / 0 / 0 | +0.004 | 0.000 | +0.58% / +0.38% / +0.29% | FAIL: retention |
| random s2 | 0.0309 | −0.001 | 0 / 0 / 0 | −0.017 | 0.000 | +0.57% / +0.40% / +0.32% | FAIL: clean F1, fabrication, retention |

Neither LoRA passed safety + guardrails. Following the pre-registered fallback, the script logged
"no config passed" and carried the larger-gain config (lr 3e-4) to the held-out folds **for the
record only**. It is not an approved arm.

## 4. Held-out folds 1–3 pooled (never used for any choice), paired vs θ_qa

| arm | abstain, absent 1 / 2 / 3 docs | abstain, answerable 1 / 2 / 3 docs | discrimination (min over k) | clean F1 | F1, all answerable | clean fabrication |
|---|---|---|---|---|---|---|
| θ_qa | 0.000 / 0.000 / 0.000 | 0.001 / 0.000 / 0.000 | −0.001 | 0.246 | 0.223 | 0.074 |
| **LoRA lr 3e-4** | **0.960 / 0.960 / 0.941** | 0.078 / 0.077 / 0.104 | **+0.837** | 0.350 | 0.307 | 0.021 |
| random s0 | 0.000 / 0.000 / 0.000 | 0.002 / 0.000 / 0.000 | −0.002 | 0.212 | 0.190 | 0.059 |
| random s1 | 0.000 / 0.000 / 0.000 | 0.002 / 0.000 / 0.001 | −0.002 | 0.238 | 0.210 | 0.071 |
| random s2 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | +0.000 | 0.223 | 0.198 | 0.081 |

Paired deltas vs θ_qa (95% bootstrap CI; ✓ = significant):

| arm | abstain Δ | clean F1 Δ | F1 Δ (all answerable) | false-abstain Δ | clean fabrication Δ |
|---|---|---|---|---|---|
| LoRA lr 3e-4 | +0.954 [+0.946, +0.961] ✓ | +0.103 [+0.085, +0.123] ✓ | +0.084 [+0.074, +0.095] ✓ | **+0.086 [+0.077, +0.096] ✓** | −0.052 [−0.068, −0.036] ✓ |
| random s0 | 0.000 | −0.034 [−0.046, −0.021] ✓ | −0.034 ✓ | 0.000 | −0.015 ✓ |
| random s1 | 0.000 | −0.008 [−0.019, +0.002] | −0.013 ✓ | 0.001 | −0.003 |
| random s2 | 0.000 | −0.024 [−0.035, −0.012] ✓ | −0.025 ✓ | 0.000 | +0.007 |

**Targeted − random**, abstention on the same 3,378 absent items: **+0.954 [+0.946, +0.961]** for all
three seeds. The random edits moved abstention on exactly zero items.

**What the held-out results show:**

- **The skill is real, not a shortcut.** Discrimination is +0.83 to +0.90 at every document count on
  every held-out fold. The Gate 2a pattern (high at one k, ~0 at the others) is gone. 3,222 of 3,378
  absent-item responses are exactly the target string `not stated in the context`.
- **It over-abstains.** 8.6% of answerable questions are now refused (292 of 3,378), with more refusals
  at 3 documents (10.4%) than at 1 (7.8%). θ_qa's own F1 on those refused items was 0.174, below its
  0.223 average. The LoRA mostly gives up where θ_qa was already weak, but not only there.
- **The clean-F1 gain needs a caveat.** On answerable items the LoRA did *not* refuse, F1 rises from
  0.253 to 0.373 with no change in answer length (~9 words), and exact match roughly doubles
  (0.039 → 0.076).
  - The references in these pools are RAFT's short answers, the same style the LoRA was trained on.
    Part of this gain is likely style matching to the reference, not better reading.
  - It has **not** been checked on the frozen eval, whose references come from a different
    generator. The frozen eval is held back as the final test (§7).
- **Fabrication falls** from 0.074 to 0.021, consistent with the model choosing to abstain instead of
  inventing.

## 5. Forgetting: why the LoRA fails, and what the random control says about edit size

The two experiments together give four LoRA points and two random-norm points (fast retention, mean of
the three domains, vs θ_qa):

| edit | ‖Δθ‖/‖θ‖ | training rows × steps | mean retention cost |
|---|---|---|---|
| Gate 2a LoRA lr 1e-4 | 0.0076 | 6,381 × 399 | +2.1% |
| Gate 2 LoRA lr 1e-4 | 0.0113 | 12,732 × 796 | +3.8% |
| Gate 2a LoRA lr 3e-4 | 0.0202 | 6,381 × 399 | +4.3% |
| Gate 2 LoRA lr 3e-4 | 0.0309 | 12,732 × 796 | +8.1% |
| random, same norm as Gate 2a lr 1e-4 | 0.0076 | — | +0.03% |
| random, same norm as Gate 2 lr 3e-4 | 0.0309 | — | +0.42% |

- **Learned edits cost forgetting roughly in proportion to their size.** Random edits cost roughly in
  proportion to the *square* of their size: 4× the norm gives ~14× the cost.
  - This is what you expect if the LoRA direction has a first-order component along the
    language-modelling loss gradient, and a random direction does not.
  - It is a pattern across six points, not a fitted law.
- **The practical consequence for Gate 3.** At ‖Δθ‖/‖θ‖ ≈ 0.03, even a *random* direction nearly
  spends the whole +0.5% retention budget. It also costs 0.01–0.03 clean F1, and two of three seeds
  fail the clean-F1 guardrail on held-out data.
  - Any edit that is to pass the approved guardrails has to be much smaller than that, around 0.01 or
    below.
  - And it needs far less first-order damage per unit of size than this LoRA has.
- **Where the comparison stands.** The full SFT that produced θ_qa cost θ_e4 +2.7% to +3.2%
  (docs/06 §5). The abstention LoRA costs θ_qa *more* than that.

## 6. Milestone checks: full held-out retention and the copy guardrail

`python -m wsp.gate2_milestone --arms lora_lr0.0003 lora_lr0.0001 random_s0`. Full retention covers
every window of both disjoint held-out sets, 35,293 windows in all. Loss change is vs θ_qa's Gate 1
per-window losses.

| arm | e4_native sec | e4_native case-law | e4_native web | v1_heldout case-law | v1_heldout web | verbatim copy (≥ 0.93) |
|---|---|---|---|---|---|---|
| LoRA lr 3e-4 | **+8.78%** | **+7.50%** | **+7.94%** | **+7.63%** | **+8.04%** | 0.970 [0.945, 0.990] PASS |
| LoRA lr 1e-4 | **+4.14%** | **+3.26%** | **+3.79%** | **+3.33%** | **+3.85%** | 0.985 [0.965, 1.000] PASS |
| random s0 | +0.58% | +0.39% | +0.25% | +0.41% | +0.27% | 0.955 [0.925, 0.980] PASS |

- **The full set confirms the fast loop measurement.**
  - On the real edits, the 512-window fast subset was within 0.09 percentage points of the full set
    (LoRA lr 3e-4: fast +8.87 / +7.55 / +7.99 vs full +8.78 / +7.50 / +7.94).
  - The second, disjoint held-out set (v1_heldout) agrees with the first.
  - The retention failure is not a measurement artefact.
- **The copy machinery is intact in every arm.** Verbatim copy passes, and induction accuracy is
  unchanged (second-copy accuracy 0.81–0.84 random-token and 0.85–0.86 natural, vs θ_qa's 0.82 and 0.86).
- **An independent sign that the reading gain is not only reference style.** The Phase-0 C2 probe
  asks a question about a code planted in a passage, and has nothing to do with RAFT's references.
  It rose from θ_qa's **0.05 [0.02, 0.08]** to **0.22 [0.165, 0.28]** (lr 3e-4) and 0.195 (lr 1e-4).
  The random edit stayed at 0.06. This is a small probe (n=200), but it is evidence that the LoRA
  also improved extracting an answer from context, not only matching the answer style of §4.

## 7. Gate 2 verdict

| question | answer |
|---|---|
| Is the loop instrument correct? | **Yes.** Identity, safety veto and fast-retention checks pass (§1). |
| Did the instrument catch a false positive? | **Yes.** The document-count shortcut (§2), caught by a filled answerable × absent × document-count grid before any result was reported. The grid was added after the suspicious ~100% result, not pre-registered; it is now a mandatory output. |
| Does a gradient method install the target skill? | **Yes.** Abstention 0.00 → 0.95 held-out, discrimination ≥ +0.83 at every document count, random control 0.000. |
| Within the approved guardrails? | **No.** Full held-out retention is +3.3% to +8.8% vs ≤ +0.5%, confirmed on both disjoint sets. Clean F1, fabrication and verbatim copy pass, but 8.6% of answerable questions are refused. No configuration passed. |
| Is there an approved Gate 2 arm? | **No.** |

**What this sets up for Gate 3.** The question for weight-space programming is now sharp: *can a
controlled edit get a meaningful share of this abstention gain at a much smaller retention cost than
LoRA?* The LoRA is the baseline any WSP arm must beat on the trade-off. The cheapest first experiment
needs no new training and no editor LLM:

- **Scale sweep of the saved LoRA delta.** `materialize(θ_qa, Δ_LoRA, scale=s)` for s ∈
  {0.1, 0.2, 0.3, 0.5}, measuring abstention vs retention.
  - If abstention survives at a scale where retention falls under +0.5%, a smaller edit exists in the
    LoRA's own direction.
  - If not, the abstention direction is intrinsically expensive at this model size.
- **Then layer- and module-restricted deltas.** Which tensors carry the abstention gain, and which
  carry the forgetting? This is exactly the per-tensor search space an editor LLM would control.

**Also still owed from the brief:**

- the **full-SFT** comparison arm (only LoRA has run);
- the **frozen-eval** measurement of the final arms, with the `retrieval_failure` caveat above. The
  decorrelated grid is the primary abstention measure;
- the self-recursion study, which needs an approved arm first.

## 🛑 GATE 2 ENDS HERE

Decisions needed before Gate 3:

1. **Keep the +0.5% retention guardrail as is?** Recommended: yes. Relaxing it would let the LoRA pass
   after the fact, and the whole point of Gate 3 is whether a controlled edit can do better than this
   trade-off.
2. **Gate 3 starts with the scale sweep and the layer/module attribution** (no editor LLM, a few hours
   of local GPU). The editor-LLM arms come after, once we know which knobs carry the effect. Choice of
   editor LLM is still open.

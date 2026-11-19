# Editor-LLM arm: can an LLM, choosing weight edits from dev feedback, beat the deterministic search?

**Approved 2026-11-11** ("Editor-LLM arm"). **Part A was written before any editor ran.**

## Part A — pre-registration

**Question.** This is the WSP question proper: does an LLM that designs weight edits in a closed loop,
seeing only dev results, find better edits than a fixed search grid? The deterministic reference is
the docs/13 AND-gate grid (12 configs, selected on `iter_val_0`).

**The editor.**
- **Who:** three independent runs (A, B, C). Each is a fresh agent of the same model as the
  experimenter (Claude Opus 5.5), with no project history.
- **What it gets:** a written brief and one command, `python -m wsp.editor_step`. That is the only
  interface it may use, and it must not read the repository or the run folders. The brief is the
  `BRIEF` string in `src/wsp/editor_step.py`.
- **What it does:** proposes an edit as JSON, receives dev metrics, and iterates.
- **Budget:** 18 evaluations per run, with safety vetoes counted. Invalid JSON is rejected and not
  charged.
- **Final pick:** at the end it nominates one step as its final edit. Each run's nominated edit is
  that run's result.

**Action space** (summed into one delta on θ_qa, at most 3 components):
- **`and_gate`:** layer ∈ {5, 6}, ρ ∈ {0.25, 0.5, 1}, τ ∈ {0, 1}, gain ∈ [0, 2]. These are the
  docs/13 vectors, with gain scaling the write.
- **`scaled_delta`:** scale ∈ [0, 1] × a layer/module mask of an existing gradient-trained delta:
  Gate 2 LoRA lr 1e-4 or 3e-4, or the docs/15 full fine-tune.
- **It cannot introduce new information.** Every vector was fit earlier on `edit_dev`. The editor's
  only lever is how it combines and scales them.

**Dev feedback per step** (`iter_val_0` only):
- abstention per kind, gain, discrimination, clean-F1 Δ with CI, false abstention, fabrication;
- fast retention per domain, guardrail flags, safety;
- **a dev overlap-matched pool**, built exactly as docs/14 but from `iter_val_0` seeds: 332 seeds,
  704 matched items, lexical AUROC 0.501 [0.452, 0.543];
- six example failures.

**The editor's stated objective:** among dev-guardrail-passing edits, maximise matched-pool D first
and abstention gain second.

**Gateway self-test** (before any editor ran; run `selftest`, not an arm):
- It reproduces docs/13's selected edit (L6, ρ 1, τ 0, gain 1) exactly: abstention gain +0.716,
  discrimination min 0.633, retention +0.014 / 0.002 / 0.003%, the same numbers as docs/13.
- Its dev matched-pool D is **0.139 [0.090, 0.192]**, the bar an editor must beat on dev.

**Held-out scoring, by the experimenter only, for each run's final edit:**
- `iter_val_1..3` pooled, paired against θ_qa;
- full retention on both disjoint sets, and the verbatim-copy probe;
- the held-out overlap-matched pool (docs/14) with its verdict rule;
- one norm-matched random control.

**Outcome per run:**
- **BEATS DETERMINISTIC:** the final edit passes every approved guardrail held-out (pooled), and the
  lower CI of its held-out matched D is above 0.122, the AND-gate edit's point estimate.
- **MATCHES:** it passes every guardrail and its CI contains 0.122.
- **WORSE:** anything else, including a failed guardrail or no nomination.

**Arm verdict:** the editor arm beats the deterministic search only if at least 2 of the 3 runs are
BEATS.

**Prediction, stated in advance: MATCHES.** The editor cannot add a new read. Its main lever, mixing
in scaled gradient deltas, costs retention quickly: at scale 1 that is +4% to +9%, against a budget
of 0.5%. So the best it can likely do is re-find the AND-gate edit, perhaps with a small scaled
component.

**Operational note (2026-11-12, made mid-run; it changes no rule or number).**
- **Slowdown:** three editors evaluating at once on one 12 GB GPU made each step take about 60–70
  minutes, against about 4 minutes alone.
- **Power cut:** a power cut then stopped all three runs. Steps completed by then: A 4, B 5, C 4.
  Every completed step is kept as logged. The calls in flight were lost and not charged.
- **Fix:** the gateway now holds a GPU lock, so evaluations run one at a time.
- **Resume:** the same three agents resumed with their own context intact.

## Part B — results (`wsp_runs/gate5_editor/`: `run_{A,B,C}/steps.jsonl`, `final.json`, `reports/`, `retention/`, `probes/`, `heldout_ovm_gen.json`; numbers in `gate5_editor_numbers.json`)

**Verdict: EDITOR BEATS DETERMINISTIC, by the pre-registered rule (3 of 3 runs are BEATS).** Each
editor's nominated edit passes every guardrail held-out. The lower CI of each one's held-out
matched-pool D is above the AND-gate edit's 0.122. **The prediction (MATCHES) was wrong.**

**Read this verdict narrowly (see "What this does and does not show").** All three winning edits are
the AND-gate *plus a scaled slice of a gradient-trained delta*. So the editors beat the docs/13 grid
partly because they had a larger action space. This arm does not show that an LLM beats a
deterministic search *over the same action space*.

**Scoring note.** To save time, A and C were scored while B was still editing. Scoring shared the
same GPU lock, so evaluations still ran one at a time. This changes no rule or number.

### What each editor nominated

| run | steps used (vetoes) | final step | final edit | ‖Δθ‖/‖θ‖ | dev D matched |
|---|---|---|---|---|---|
| A | 18 (2) | 18 | gate L6 ρ 1 τ 0 gain 1.15 + **LoRA 3e-4 × 0.7, layers 0–5, q/k only** | 0.0362 | 0.253 [0.184, 0.325] |
| B | 17 (1) | 17 | gate L6 ρ 1 τ 0 gain 1.1 + **full FT × 0.25, layers 0–5, all projections** | 0.0349 | 0.253 [0.190, 0.318] |
| C | 17 (1) | 16 | gate L6 ρ 0.5 τ 0 gain 1.5 + **LoRA 3e-4 × 1.0, layers 0–8, q only** | 0.0298 | 0.239 [0.171, 0.309] |

**All three converged independently on the same recipe:**
- the layer-6 gate at τ 0, written about as hard as the per-tensor safety cap allows;
- a slice of a gradient delta restricted to the early layers, before the gate, mostly in attention.

**What each editor reported learning, in its own words, and the logs agree:**
- the gate alone mostly learns overlap (dev D ≤ 0.15);
- the layer-5 gate is weak, and two gates interfere;
- a trained delta alone barely abstains, but on top of the gate it strongly raises D;
- the cheapest D per unit of retention is early attention.

### Held-out `iter_val_1..3` pooled, paired against θ_qa (3,378 absent and 3,378 answerable items)

| | run A | run B | run C | AND-gate edit (docs/13) |
|---|---|---|---|---|
| abstention Δ on absent items | +0.931 [+0.922, +0.939] | +0.898 [+0.887, +0.908] | +0.845 [+0.832, +0.857] | +0.737 |
| abstain abs1 / abs2 / abs3 | 0.950 / 0.937 / 0.905 | 0.885 / 0.905 / 0.903 | 0.928 / 0.891 / 0.715 | — |
| false abstention ans1 / ans2 / ans3 | 0.090 / 0.080 / 0.078 (Δ +0.082) | 0.041 / 0.067 / 0.106 (Δ +0.071) | 0.089 / 0.082 / 0.083 (Δ +0.084) | — |
| discrimination min (k = 1..3) | +0.827 | +0.798 | +0.631 | — |
| clean F1 Δ | +0.029 [+0.015, +0.045] | **+0.071 [+0.055, +0.087]** | +0.013 [−0.001, +0.027] | −0.005 |
| clean fabrication Δ | −0.029 [−0.044, −0.015] | −0.031 [−0.048, −0.014] | −0.031 [−0.046, −0.016] | — |
| worst retention (5 sets) | +0.35% | +0.33% | +0.43% | +0.011% |
| verbatim copy | 0.955 | 0.960 | 0.955 | — |
| **all guardrails held-out** | **pass** | **pass** | **pass** | pass |

- **Disclosure for run C, fold 1:** its clean-F1 guardrail fails on that fold alone (Δ −0.004). The
  pre-registered test is pooled, where it passes (lower CI −0.001 ≥ −0.02). C's abs3 rate is also
  clearly weaker (0.715).
- **Random controls** (norm-matched, seed 0, one per run):
  - abstention 0.000 on every fold;
  - clean F1 −0.002 to −0.006;
  - copy 0.960;
  - retention +0.17% to +0.65%.
- **The random controls cost more retention than the edits.** A's and B's controls exceed the 0.5%
  budget on e4 sec (+0.65%, +0.62%). Like docs/15, retention cost depends on direction, not size.
  Here the editors' directions were cheaper than random.

### Overlap-matched pool (held-out, docs/14 verdict rule)

| model | D on pool v2 | **D matched** | survives | D deleted-only | abstain del1 / del2 | abstain ans1 / ans2 | matched-ans F1 Δ | verdict | outcome vs 0.122 |
|---|---|---|---|---|---|---|---|---|---|
| AND-gate edit | +0.720 | +0.122 [+0.097, +0.150] | 17% | +0.061 | 0.088 / 0.118 | 0.034 / 0.051 | −0.013 | OVERLAP-DOMINATED | (the bar) |
| **run A** | +0.859 | **+0.229 [+0.192, +0.264]** | 27% | +0.148 | 0.233 / 0.279 | 0.116 / 0.102 | +0.006 (n.s.) | PARTIAL | **BEATS** |
| **run B** | +0.841 | **+0.184 [+0.154, +0.217]** | 22% | +0.110 | 0.145 / 0.217 | 0.054 / 0.090 | +0.045 [+0.031, +0.058] | PARTIAL | **BEATS** |
| **run C** | +0.824 | **+0.233 [+0.196, +0.269]** | 28% | +0.155 | 0.247 / 0.281 | 0.112 / 0.106 | −0.003 (n.s.) | PARTIAL | **BEATS** |
| LoRA lr 3e-4 | +0.882 | +0.263 [+0.224, +0.300] | 30% | +0.178 | 0.271 / 0.295 | 0.107 / 0.104 | +0.084 | PARTIAL | fails retention |
| full FT lr 1e-5 | +0.862 | +0.254 [+0.216, +0.291] | 29% | +0.170 | 0.293 / 0.312 | 0.139 / 0.127 | +0.080 | PARTIAL | fails retention |

**Dev → held-out shrinkage of matched D:**

| run | dev | held-out |
|---|---|---|
| A | 0.253 | 0.229 |
| B | 0.253 | 0.184 |
| C | 0.239 | 0.233 |

B's edit shrank most, but its CI stays clear of 0.122. Choosing among 17–18 dev looks inflated the dev
numbers somewhat. It did not create the effect.

### Comparison of every arm (held-out, pooled)

| arm | abstain Δ | clean F1 Δ | worst retention | matched-pool D | verdict | passes all guardrails |
|---|---|---|---|---|---|---|
| LoRA lr 3e-4 | +0.925 | +0.096 | +8.87% | +0.263 | PARTIAL | no |
| full FT lr 1e-5 | +0.960 | +0.105 | +5.86% | +0.254 | PARTIAL | no |
| AND-gate edit (deterministic grid) | +0.737 | −0.005 | +0.011% | +0.122 | OVERLAP-DOMINATED | yes |
| **editor A** | +0.931 | +0.029 | +0.35% | +0.229 | **PARTIAL** | **yes** |
| **editor B** | +0.898 | **+0.071** | +0.33% | +0.184 | **PARTIAL** | **yes** |
| **editor C** | +0.845 | +0.013 | +0.43% | +0.233 | **PARTIAL** | **yes** |

### Prediction check

- **Wrong:** the predicted outcome was MATCHES, and the result was BEATS in all three runs.
- **Right on the mechanism, wrong on its size:**
  - Mixing in a scaled gradient delta was indeed the main lever.
  - Restricted to the early layers and a few projections, it cost far less retention than the
    prediction assumed. Scale 1 on all layers costs +4% to +9%. The editors' slices cost +0.3% to +0.4%.
  - The prediction extrapolated linearly from full-delta costs. The editors instead found that
    early-layer slices buy the most matched D per unit of retention cost. Run A, dev: layers 6–11 ×
    0.2 gave D 0.162 at +0.28%; layers 0–5 × 0.17 gave D 0.199 at +0.38%.

### What this does and does not show

1. **First arm that is PARTIAL on the matched pool while inside every guardrail.**
   - Docs/15 concluded that "nobody has solved" getting abstention beyond overlap inside the
     retention budget. The editors' edits are the first to reach PARTIAL there.
   - They keep 70–90% of the gradient arms' matched D (0.18–0.23 vs 0.25–0.26) at about 1/15 to
     1/27 of the forgetting (+0.3–0.4% vs +5.9–8.9%).
   - None is BEYOND OVERLAP. About 22–28% of discrimination survives when overlap is uninformative.
2. **This is partly fine-tuning, not pure weight-space programming.**
   - Each winning edit carries a slice of a gradient-trained delta. The editors' own logs show the
     gate alone tops out at dev D ≤ 0.15, so most of the gain beyond overlap comes from that slice.
   - What the editors added is the *selection*: which layers and modules of the trained delta to
     keep, at what scale, and on top of which gate.
   - So this is "targeted partial fine-tuning, chosen by closed-loop editing". It is not "a hand-built
     edit with no training".
3. **The comparison is confounded by the action space.** The deterministic reference (docs/13 grid)
   could not use the trained deltas. Whether the *LLM* matters, rather than just the *larger menu*,
   is untested. A simple deterministic search over the same menu might find the same "gate +
   early-attention slice" with the same 18-evaluation budget, for example a small grid on
   layers 0–5 × module set × scale.
4. **The convergence is evidence the result is real, not luck.** Three independent agents with no
   shared context reached the same structure. All three also held up on held-out data.
5. **The costs that remain:**
   - False abstention rises by +0.07 to +0.08 (about 1 in 12 answerable questions is now refused).
   - Clean F1 gains are small, except for B, which used the full-FT slice.

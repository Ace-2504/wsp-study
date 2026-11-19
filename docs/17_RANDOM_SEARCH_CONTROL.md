# Same-menu control: does uninformed random search over the editors' action space do as well?

**Approved 2026-11-15** ("i will do optional then self recursion then final frozen eval"). **Part A was
written before any sample was drawn or evaluated.**

## Part A — pre-registration

**Question.** In docs/16, all three LLM editors beat the deterministic AND-gate grid. However, the
editors could also mix in scaled slices of gradient-trained deltas, and the grid could not. This
control separates the two explanations:

- **(a) the larger menu** is enough: any reasonable search over it would win;
- **(b) the LLM's choices** matter.

**The control: seeded uniform random search** over exactly the editors' action space.
- **Gateway:** the same `wsp.editor_step.evaluate`.
- **Dev feedback:** the same `iter_val_0` and dev matched pool.
- **Budget:** the same 18 evaluations per run, with vetoes counted.
- **No adaptation:** the search does not look at any result while it draws.
- **Why uniform:** this is the standard uninformed baseline for an LLM optimiser. A hand-designed
  grid is not possible here without leakage, because the experimenter has already seen the editors'
  answer (layer-6 gate plus an early-attention slice). Any grid I drew now would encode it.

**Sampling distribution** (`src/wsp/random_search.py`, fixed before any draw). For each step:
- **Number of components:** uniform over {1, 2, 3}.
- **Component type:** each is `and_gate` with probability 0.5, otherwise `scaled_delta`.
- **`and_gate`:**
  - layer uniform over {5, 6}; ρ uniform over {0.25, 0.5, 1}; τ uniform over {0, 1};
  - gain uniform on [0, 2], rounded to 0.01.
  - If two gates land on the same layer, which the gateway forbids, the whole action is redrawn.
- **`scaled_delta`:**
  - source uniform over the 3 trained deltas;
  - scale uniform on [0, 1], rounded to 0.01;
  - layers: a contiguous range [a, b], with (a, b) drawn uniformly from all 78 pairs a ≤ b in 0..11;
  - modules: each of the 9 module names kept independently with probability 0.5, redrawn if empty.
- **Invalid actions:** any action `build_delta` rejects is redrawn and not charged, as for the
  editors.
- **Seeds:** run R1 uses seed 1, R2 seed 2, R3 seed 3.

**Nomination: the editors' stated objective, applied mechanically.**
- Among steps that pass the safety gate and every dev guardrail (`guardrails_pass`, including fast
  retention), pick the one with the highest dev matched-pool D.
- Ties go to the larger abstention gain.
- If no step passes, there is no nomination.

**Held-out scoring and outcome per run:** identical to docs/16 (`wsp.editor_report`). That means
`iter_val_1..3` pooled, full retention, copy, one norm-matched random control, and the held-out
matched pool. The outcome is BEATS / MATCHES / WORSE against 0.122, and WORSE includes "no
nomination".

**Sequential design** (to keep the cost to about 2–3 h where possible):
1. Run R1.
   - **If R1 is BEATS:** stop. One uninformed run of 18 draws found a winning edit, which shows the
     menu alone is enough (explanation a). The docs/16 result is then not evidence that the LLM matters.
   - **If R1 is not BEATS:** run R2 and R3.
2. With all three runs, random search counts as matching the editors only if at least 2 of 3 are
   BEATS, the same bar the editors met. Otherwise the conclusion is: with this budget, the LLM's
   choices add value over random search over the same menu (explanation b).

**Prediction, stated in advance: R1 is not BEATS, and random search scores 0 or 1 of 3.**
- Uniform draws usually include later layers, MLPs, or high scales of the trained deltas, which fail
  retention (scale 1 costs +4–9%).
- Draws that skip the layer-6 gate with τ 0 and high gain give low matched D.
- The winning region (L6 gate, τ 0, gain ≥ 1, plus a small early-layer slice) is a small fraction of
  the space for 18 draws.

**Limitation, stated in advance.** Random search is the weakest sensible search. If the editors beat
it, that does not show they beat a well-designed deterministic optimiser (Bayesian optimisation,
coordinate descent) on this menu.

## Part B — result (R1 only; sequential stop)

**Verdict: MENU SUFFICES.** The first uninformed run, R1 with seed 1, nominated an edit that is
**BEATS DETERMINISTIC** held-out. Under the Part A rule, R2 and R3 were therefore not run. The
docs/16 result ("editors beat the deterministic grid 3/3") is **not evidence that the LLM's choices
matter**: an 18-draw uniform search over the same menu also beat the grid. **My prediction (R1 not
BEATS) was wrong.**

**The search** (`wsp_runs/gate5_editor/run_R1/`):
- 18 draws, 2 safety vetoes (steps 7 and 12), 11 of 18 passing every dev guardrail.
- Dev failures were the expected kind: retention (steps 1, 4, 16), and fabrication or validity on gate-only
  draws with low gain (steps 2, 3).
- Nomination was mechanical: **step 10**, the passer with the highest dev matched D (0.207 [0.138, 0.275]).

**The nominated edit (step 10).** It does **not** use the editors' pattern:
- no L6 gate; an **L5** AND-gate, ρ 0.25, τ 0, gain 1.57;
- the full-FT (lr 1e-5) delta ×0.55 on layers **3–11**, modules q, k, up, norm;
- the LoRA (lr 1e-4) delta ×0.09 on layers 3–4 (q, k, o, up, down, embed).

So "L6 gate plus an early-layer slice" is not the only winning region.

**Held-out (`iter_val_1..3` pooled, n = 3,378; `gate5_random_numbers.json`):**

| | R1 final | its norm-matched random control |
|---|---|---|
| ‖Δθ‖/‖θ‖ | 0.0308 | 0.0308 |
| Abstention Δ | **+0.754** [0.740, 0.768] | +0.000 |
| Discrimination (min fold) | 0.575 | 0.000 |
| Clean F1 Δ | **+0.048** [0.035, 0.062] | +0.008 (n.s.) |
| False abstention Δ | +0.075 [0.067, 0.085] | −0.000 |
| Fabrication Δ | −0.045 [−0.060, −0.031] | −0.012 |
| Worst full retention | +0.40% (fineweb-edu) | +0.46% (sec) |
| Verbatim copy | 0.96 [0.93, 0.985] | 0.96 |
| **Matched-pool D** | **0.172 [0.136, 0.207]**, PARTIAL | — |

Every held-out guardrail passes. The lower CI of matched D, 0.136, is above the AND-gate's 0.122.

**Descriptive comparison with the editors** (not pre-registered; one random run against three editor
runs, so no test):
- R1's matched D of 0.172 is below editors A (0.229) and C (0.233), and close to B (0.184). Its CI
  overlaps B's and only just reaches the lower ends of A's and C's.
- Abstention gain (+0.75 against +0.85–0.93) is lower. Clean F1 (+0.048) sits inside the editors' range.
- One draw cannot tell whether the editors are *somewhat* better than random search. It only shows
  that beating the grid needs no LLM.

**What this means for the study.** In docs/16, the deciding factor was the **action space**:
mixing gradient-trained delta slices into a gated edit. The LLM was not. Every winner on this menu
(three editors, one random run) is partly fine-tuning. **The only pure weight-space edit inside every
guardrail is still the docs/13 AND-gate**, with matched D 0.122 (overlap-dominated).

**Limitations.**
- A single random run: the probability that a random run beats the grid is not estimated.
- The norm-matched random control for R1 reaches +0.46% sec retention. That is inside the 0.5%
  budget but close to it, so the retention margin of edits at this norm is thin.

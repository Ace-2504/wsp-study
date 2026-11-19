# Gate 3 — pre-registration (written before any Gate 3 edit runs)

**Approved 2026-11-10** (after Gate 2):
- keep the +0.5% retention guardrail unchanged;
- start Gate 3 with the scale sweep and the layer/module attribution;
- editor-LLM arms come after.

**Gate 3 question:** can a *controlled* weight-space edit get a meaningful share of the Gate 2 LoRA's
abstention gain while staying inside every approved guardrail? (On held-out data the LoRA gave
abstention +0.954, at a held-out retention cost of +7.5% to +8.8%.)

Gate 3 is split in two:

- **3a (this plan): deterministic search, no LLM.** It edits the weight-space knobs an editor LLM
  would later control: global scale, which module types, which layers. It is both the first
  experiment and the **non-LLM search baseline** that any later LLM controller must beat on the
  same evaluation budget.
- **3b (later):** editor-LLM arms, once 3a shows which knobs carry the effect.

## Gate 3a design

- **Source deltas.** The two saved Gate 2 LoRA deltas (`wsp_runs/gate2/deltas/lora_lr{0.0001,0.0003}.safetensors`).
  - Nothing is retrained. Every Gate 3a edit is a scaled and/or masked copy of these deltas, applied
    to an in-memory θ_qa.
- **Phase A, scale sweep.** For each source delta, θ_qa + s·Δ with s ∈ {0.05, 0.1, 0.2, 0.3, 0.5}
  (10 configurations; s = 1.0 is Gate 2).
- **Phase B, attribution.** Δ_lr3e-4 at scale 1.0, restricted to one of 6 groups:
  - module type: attention (q, k, v, o) or MLP (gate, up, down);
  - layer band: L0–3, L4–7 or L8–11;
  - per group, the report gives abstention gain, retention cost and efficiency = abstention Δ / mean
    retention cost.
- **Phase C, efficient subsets** (the rule is fixed now; the groups are picked by Phase B's numbers).
  1. Rank the groups whose abstention Δ is significantly > 0 by efficiency.
  2. Take the union of the top-k groups, k ∈ {1, 2, 3}.
  3. Apply each union at scale s ∈ {0.25, 0.5, 1.0}, skipping configurations already run.
- **Every configuration** goes through the same pipeline as Gate 2:
  - the safety gate first;
  - then `iter_val_0` (the dev fold): abstention on abs1–3, discrimination per document count,
    clean F1 and fabrication paired vs θ_qa, false abstention, and fast retention;
  - the guardrails are the approved ones, with thresholds unchanged (`src/wsp/config_wsp.py`).
- **Validity condition (shortcut check, not a guardrail).** discrimination min over k must be ≥ half
  the abstention gain. An edit that abstains by document count rather than content fails it.
- **Selection** (dev fold only):
  - among configurations passing safety, all approved guardrails and the validity condition, pick
    the **largest abstention gain**;
  - **if none passes, Gate 3a stops**, reports the abstention-vs-retention frontier, and runs no
    held-out evaluation.
- **Report for a selected edit:**
  - held-out folds `iter_val_1..3` pooled, paired vs θ_qa;
  - 3 random controls matched in rank and per-tensor norm to the selected edit;
  - the milestone checks: full retention on both disjoint held-out sets, plus the copy and
    induction probes.
- **What counts as a positive result.** A selected edit that passes every guardrail on held-out data
  (full retention included) **and** beats all three matched random controls on abstention.
  Anything less is reported as negative.

The frozen eval is still held back for the final test.

# Deliverable 4 — Implementation Architecture

> **Revised — see [`05_LINEAGE_REVISION.md`](05_LINEAGE_REVISION.md).** The component design still
> stands. Changes: there are now **two** hash-locked originals (θ_e4 and θ_qa); the EditEngine gains a
> `task_vector` method (θ_e4 + Σ α_m·τ_m); and Gate 1 now has a concrete pass criterion: reproduce
> θ_qa's clean token-F1 0.229 / EM 0.038 / fabrication 0.062 (Revision 2 §7). Code lives under
> `../src/`; verbatim imports under `../vendor/`.

**For:** the WSP study of Deliverable 3, on the 125M `slm-125m` family.
**Design stance:** reuse the repo's stack (Python 3.12, stock HF `LlamaForCausalLM`, config-as-truth,
Modal optional). A 125.8M model loads locally in fp32 (~0.5 GB), so **the edit-eval loop runs on one
machine** (CPU-feasible; a single consumer GPU is ample). The external LLM editor is the only cloud
dependency. **No original weight is ever mutated in place.**

---

## 1. Component map (the brief's 10 components → concrete modules)

Proposed package under `wsp/` (code, separate from these design docs):

```
wsp/
  engine/
    checkpoints.py     # (1)  CheckpointStore — immutable original, per-experiment/iteration copies
    evalset.py         # (2)  EvalManager — frozen splits, decontam, leakage guard
    introspect.py      # (3)  Introspector — activations, gradients, weight stats, attributions
    programmer.py      # (4)  WeightProgrammer — the editor LLM; emits a structured EditProposal
    editor.py          # (5)  EditEngine — applies an EditProposal to a *copy*; LoRA/mask/subspace
    safety.py          # (6)  SafetyGate — validates proposals + resulting weights; veto power
    evaluate.py        # (7)  evaluate(model) -> MetricReport  (QA + retention + reliability)
    tracking.py        # (8)  ExperimentLog — append-only JSONL, per-item results, seeds, hashes
    loop.py            # (9)  the iterative feedback controller (edit→eval→feedback→propose)
    compare.py         # (10) diff/rollback: metric deltas, ‖Δθ‖, per-task regression, restore
  schemas.py           #      EditProposal / MetricReport / IterationRecord dataclasses
  run_wsp.py           #      CLI entrypoint; config-driven, one arm per invocation
  config_wsp.py        #      single source of truth (extends the repo's config.py style)
```

This mirrors the repo's existing convention (`config.py` as truth, `train_core.py` as engine,
`eval.py` as scorer, `observations*/` as append-only telemetry).

---

## 2. Data flow (one iteration)

```
CheckpointStore.load(parent)                      # immutable parent weights → working copy θ_k
        │
        ▼
Introspector.snapshot(model, edit_dev_failures)   # activations on failing QA items, per-param grads,
        │                                          # weight norms, logit-lens / causal-trace targets
        ▼
WeightProgrammer.propose(introspection, history)  # editor LLM → EditProposal (WHERE + HOW + budget)
        │                                          # structured JSON: target modules, method, magnitude
        ▼
SafetyGate.validate_proposal(proposal)            # bounds, allowed modules, magnitude cap → accept/veto
        │ accept
        ▼
EditEngine.apply(working_copy, proposal)          # produces θ_{k+1} on a COPY (LoRA merge / masked
        │                                          # gradient step / subspace delta)
        ▼
SafetyGate.validate_weights(θ_{k+1})              # NaN/Inf, ‖Δθ‖ cap, activation-explosion probe,
        │ accept                                   # perplexity-sanity on a tiny canary set → accept/veto
        ▼
evaluate(θ_{k+1}) -> MetricReport                 # QA + retention + reliability, all with CIs
        │
        ▼
compare(parent, θ_{k+1})                          # metric deltas, per-task regression, ‖Δθ‖
        │
        ├── better & safe → CheckpointStore.commit(θ_{k+1}); feed report back to programmer
        └── worse/unsafe → rollback to parent; feed the FAILURE (with reason) back to programmer
```

The **iterative feedback mechanism (9)** is this loop closed: `MetricReport` + `compare` result +
the specific items that regressed become the *next* `WeightProgrammer.propose` input. That is what
makes it "programming," not one-shot editing — the editor reasons over *why the last edit
failed/helped* and adjusts.

---

## 3. Component contracts (the interfaces that matter)

**(1) CheckpointStore.** `original` is written once, hashed (SHA-256), and opened **read-only**
thereafter; a CI-style check asserts the hash is unchanged at the start of every run (the repo already
practices "v1 verified untouched"). Every experiment gets `exp_id`; every iteration gets `θ_k` saved
as a full checkpoint *or* a compact delta against its parent. Lineage is a tree, not a line, so
rollback = "re-point to parent," never a destructive restore.

**(3) Introspector** — what the editor is *allowed to know* (the brief's "what information does the
programmer need?"). Produces, on demand and on the `edit-dev` failure set only:
- weight statistics per module (norms, spectra, dead/saturated units);
- **activations** captured via forward hooks on failing QA prompts (mean/var, which neurons fire on
  the question vs the answer span);
- **gradients** of the QA loss w.r.t. parameters (`torch.autograd`), i.e. "which weights, if nudged,
  most reduce the question→answer loss" — the single most informative signal;
- optional **causal attribution** (activation patching / logit-lens) to localise where the
  question→span mapping *would* live.
These are serialised into a compact, token-budgeted summary (not raw tensors) for the editor LLM.

**(4) WeightProgrammer** — the external editor LLM (a capable model such as the one running this
project). It receives the architecture description, the introspection summary, the eval-failure
digest, and the edit history; it returns a **structured `EditProposal`**, never free numbers:
```
EditProposal = {
  rationale: str,                       # why this edit, tied to the evidence it saw
  method: "masked_grad"|"lora"|"subspace"|"direct_delta",
  targets: [{module: str, scope: "weight"|"neuron"|"rank", index?: ...}],
  magnitude: float,                     # bounded; SafetyGate enforces the cap
  expected_effect: str,                 # a prediction, logged, later scored for calibration (RQ4)
}
```
The programmer **proposes where and how**; the numerical values come from a grounded mechanism
(gradient / search), except in the R6 `direct_delta` probe where it emits numbers directly (expected
to fail — that is the point).

**(5) EditEngine** realises each `method` on a copy: `lora` = train/merge adapters on the masked
targets; `masked_grad` = one or few bounded gradient steps restricted to the proposal's mask;
`subspace` = ES/CMA-ES over a low-rank basis the proposal names; `direct_delta` = add the editor's
numbers. **Tied embeddings are handled explicitly** — an edit to the embedding matrix is flagged as
affecting both read and predict paths.

**(7) evaluate** is the Deliverable-1 §6 panel as one in-process callable returning a `MetricReport`
with per-metric point estimate + bootstrap CI + per-item results. Must be **deterministic** (greedy
decode, fixed seed) and fast enough to call every iteration.

---

## 4. Safety layer (the brief's safeguards, made mechanical)

`SafetyGate` has veto power at two checkpoints (proposal-time and weight-time). It guards against each
named risk:

| Risk (brief Phase 5) | Mechanism |
|---|---|
| Uncontrolled parameter modifications | per-module **magnitude cap** on ‖Δθ‖ (relative Frobenius); proposals over cap are clipped or vetoed |
| NaN / infinite weights | post-edit `isfinite` assertion over all params → veto + rollback |
| Numerical instability | spectral-norm / max-abs-weight bounds per module; bf16↔fp32 consistency check |
| Exploding activations | forward-pass **canary**: activation max/var on a fixed probe batch must stay within a band of the original |
| Severe performance regression | `compare` fails the arm if the **retention panel breaches pre-registered thresholds** (SEC bpb +2%, verbatim-copy ≥70%, CaseHOLD ≥0.46) → auto-rollback |
| Evaluation leakage | `EvalManager` enforces disjoint `edit-dev`/`iter-val`/`final-test`; the editor is *structurally* denied access to `iter-val`/`final-test`; QA passages decontaminated (13-gram) against **both** corpora |
| Overfitting to benchmark questions | `iter-val` is **rotated** each iteration; `final-test` is touched exactly once; gains must track across both (Deliverable 3 §6) |

Every veto is logged with its reason and fed back to the programmer as a hard constraint for the next
proposal — safety failures are *training signal*, not silent drops.

---

## 5. Reproducibility & tracking (8)

- **Config-as-truth:** `config_wsp.py` pins seeds, decoding params, split hashes, magnitude caps,
  thresholds, editor model id + sampling params. One file reproduces an arm.
- **Append-only `ExperimentLog`** (JSONL, mirroring `observations*/`): one `IterationRecord` per
  iteration with the proposal, the safety verdicts, the full `MetricReport`, per-item results,
  ‖Δθ‖, parent hash, and child hash. Nothing is overwritten.
- **Determinism:** fixed seeds; greedy decode for scored QA; `torch.use_deterministic_algorithms`
  where feasible. The editor LLM is stochastic, so its **proposal and the resulting metrics are both
  logged** — an arm is reproducible given the logged proposals even if a fresh editor call would
  differ.
- **Checkpoint tree** under `wsp_runs/<exp_id>/iter_<k>/`, original under `wsp_runs/_original/`
  (read-only, hash-locked). All gitignored like the repo's other large artifacts.

---

## 6. Cost & placement

The loop is cheap: a 125M model edit + the Deliverable-1 eval panel run in minutes on one GPU (or
slower on CPU). The dominant cost is **editor LLM API calls** (one `propose` per iteration) and any
`masked_grad`/`lora` micro-training (seconds on Modal L4, matching `SFT_PLAN.md §9`'s ~$0.19/arm).
Budget is therefore not a constraint; wall-clock and the honesty of the eval are. Modal is used only
for the occasional GPU micro-train and for regenerating the external bits-per-byte baseline if e2/e4
are targeted (audit §7).

---

## 7. Build order (so Deliverable 5 can start safely)

1. `CheckpointStore` + hash-locked original + `evaluate()` panel + `EvalManager` with frozen,
   decontaminated splits. **No editing yet** — first prove the harness reproduces A0 (the base's
   0% QA / 76.5% copy / SEC-win numbers) deterministically.
2. `SafetyGate` + `EditEngine` with the **random-edit (A1)** and **LoRA (A3)** arms — these need no
   editor LLM and validate the whole pipeline + safety + rollback end to end.
3. `Introspector` + `WeightProgrammer` + `loop.py` — the WSP arms (A5–A8).
4. Ablations (Deliverable 3 §7) and the self-recursion study (§6).

Gate 1 is the analogue of this family's Phase-0 discipline: **prove the instrument before trusting any
result it reports.**

---

## 🛑 DELIVERABLE 4 ENDS HERE

Architecture only. Implementation (Deliverable 5) begins after the plan and these safeguards are
approved, starting from Gate 1 above.

# Revision 2 — Retarget to the `e4 → slm-125m-sft` QA lineage

**Supersedes** the *target-model* parts of Deliverables 1, 3 and 4 (each carries a banner pointing
here). The feasibility verdict (Deliverable 2) is unchanged. Nothing has been edited or trained;
every number below is transcribed from saved artifacts in `data/reference_results/` or was measured
by a read-only check on the imported data.

---

## 1. Why the target changed

Revision 1 targeted `slm-125m-extended`, a base completer that answers question-cued retrieval at
0/200. A QA-tuned checkpoint **does** exist in the same 125M family, with its own pretrained base
published next to it:

```
slm-125m-base → extended → e2 → e4  ──(full FT, 15k grounded QA pairs)──►  slm-125m-sft
                                     ├─(full FT, 10k RAFT)───────────────►  slm-125m-raft
                                     └─ slm-125m-sft ─► -sft-dpo / -sft-rlaif
```

All are on the HF Hub (`Ace-2504/…`), with identical architecture and tokenizer (verified from each
`config.json`: `LlamaForCausalLM`, 12 L, 768 d, vocab 16,384, ctx 1,024, tied embeddings). The
training recipe is `vendor/finetuning/ft_config.py::M125` (lr 3e-5→3e-6, 3 epochs, effective batch
16, assistant-only loss, no BOS).

**The roles in this study:**

| Symbol | Checkpoint | Role |
|---|---|---|
| θ_e4 | `Ace-2504/slm-125m-e4` | **Original pretrained model.** Untouched reference for pretraining-knowledge forgetting. |
| θ_qa | `Ace-2504/slm-125m-sft` | **Original QA model.** Primary WSP edit target; untouched reference for QA regressions. |
| τ | θ_qa − θ_e4 | **The QA task vector** — a full 125.8M-dim weight delta that encodes "how e4 became a QA model." |
| — | `-raft`, `-sft-dpo`, `-sft-rlaif` | Sibling references with published scores on the same instrument. |

HF revisions are pinned by commit SHA at download time (Gate 1) and hash-locked thereafter.

## 2. Why this is a better scientific setting

1. **The starting point is not zero.** θ_qa answers (judged correctness 0.261). WSP can now be tested
   where the editing literature has at least some traction: *sharpening or repairing an existing
   skill* (fabrication, groundedness, abstention, distractor robustness) instead of installing one
   from nothing.
2. **The "install" question still gets answered, and more cheaply.** For e4 → QA, the full-SFT
   reference arm (A4 in Deliverable 3) **already exists**: it is θ_qa. Only the LoRA-from-e4 arm needs
   training.
3. **τ gets around the dimensionality wall.** Deliverable 2 §2 showed that gradient-free or
   LLM-authored search over 125.8M raw weights can't work. But the QA skill already exists as a
   direction in weight space. Per-module **task-vector surgery**,
   θ(α) = θ_e4 + Σ_m α_m · τ_m with one coefficient α_m per weight matrix, is a search over about
   110 numbers (12 layers × 9 matrices + embeddings + final norm). That size is exactly where CMA-ES
   and LLM-guided proposals are known to work (Akiba et al. 2024 ran CMA-ES over merge coefficients
   of this order). It is also interpretable: the α profile shows **which modules carry the QA skill
   and which carry the forgetting.** This is the most defensible "LLM programs the weights" variant
   available. The LLM proposes and reasons about α, which is a small, meaningful parameter space,
   while every individual weight value comes from an established operation (task arithmetic).

## 3. Baseline: published numbers on the frozen instrument

Instrument: `data/eval/eval_messages.jsonl`, SHA-256 `16e12b23…` (byte-identical to the `eval_sha256`
in the reference manifests). 500 grounded-QA items × 4 conditions (clean / realistic /
retrieval_failure / closed_book), same items as the anonymised paper release
`data/eval/eval_set.jsonl`. Decoding: greedy, 160 new tokens, no BOS. Judge: Gemini rubric, blind.
95% bootstrap CIs.

**Clean condition** (`data/reference_results/REPORT.md`, SET1):

| Model | Judged correctness | Groundedness | Token-F1 | Exact match | Fabrication ↓ | False-abstain ↓ |
|---|---|---|---|---|---|---|
| `slm-125m-base` (v1, 3-shot), **not e4** | 0.000 [0.000, 0.000] | 0.068 | 0.019 | — | 0.110 | 0.000 |
| **slm-125m-sft** | **0.261** [0.228, 0.293] | 0.456 | **0.229** [0.206, 0.251] | **0.038** [0.022, 0.054] | **0.062** [0.042, 0.084] | 0.000 |
| slm-125m-raft | 0.049 [0.033, 0.067] | 0.108 | 0.078 | — | 0.002 | 0.836 |

Rubric-10 leaderboard (`STATS_p3a_baselines.md`): 125M-SFT **3.60**, 125M-RLAIF 3.69, 125M-DPO 0.90,
125M-RAFT 0.81, 125M-base 0.06. External reference points at larger scale: Qwen2.5-0.5B-Instruct 7.77,
SmolLM2-1.7B-Instruct 8.98.

**Correction (found at the start of Gate 1):** the published "125M base" row was produced with
`Ace-2504/slm-125m-base` (v1), as the manifest of `base-125m.json` and `BASES` in
`vendor/finetuning/eval.py` both show. It is **not** θ_e4. **θ_e4's QA zero point has never been
measured.** Gate 1 measures it with the same 3-shot protocol.

**Gaps in the published baseline that Gate 1 must close:** the per-item reference results for
θ_qa exist **only for the clean condition**, so its abstention on `retrieval_failure`, its distractor
robustness (`realistic`) and its closed-book behaviour are **unmeasured**. Pretraining-retention
(forgetting from e4 to sft) has also **never been measured** for this lineage.

## 4. New finding: train/eval n-gram overlap (measured during import)

A 13-gram overlap scan of the training pools against the eval set's clean passages:

| Pool | n | Shared pair IDs | Pairs with any shared 13-gram | ≥10% of own 13-grams shared | ≥50% shared |
|---|---|---|---|---|---|
| `qa_sft.jsonl` (θ_qa's training data) | 15,000 | 0 | 483 (3.2%) | 134 | **36** |
| `raft.jsonl` | 10,000 | 0 | 1,823 (18.2%) | 195 | 9 |

Most of the overlap is shallow boilerplate. The most common shared gram is a standard Kentucky
appellate header, *"unpublished Kentucky appellate decisions rendered after January 1, 2003 may be…"*.
A minority, though, is substantial passage overlap. **Consequences:**

- **WSP `edit-dev` excludes all 483 overlapping `qa_sft` pairs** (conservative quarantine), so no
  editing decision is ever made on text that appears in the final test.
- **The published θ_qa scores may be slightly optimistic.** Gate 1 will quantify this rather than
  assume it, by splitting eval items into overlapping and non-overlapping passages and reporting the
  score gap with a CI. With ≤134 deeply overlapping training pairs out of 15,000, I expect the effect
  to be small, but it should be measured, not asserted.

## 5. Revised experiment (replaces Deliverable 3 §4's arm list)

**Part I: Sharpen.** Edit θ_qa. This is the primary study.

| Arm | Method | Question |
|---|---|---|
| S0 | θ_qa untouched | zero point |
| S1 | Random edit, magnitude-matched to S5/S6 | Does targeting matter? (RQ2) |
| S2 | Continued LoRA-SFT on `edit-dev`, same data and compute budget | **The fine-tuning bar** (RQ6) |
| S5 | WSP, introspection-targeted gradient edit (the LLM picks modules from activation/gradient evidence on failures) | Defensible WSP core |
| S6 | **WSP, task-vector surgery**: LLM-guided (vs. CMA-ES) search over per-module α | The tractable "LLM programs weights" variant |
| S7 | Hybrid: S6's α profile selects modules, LoRA fills them | Complementarity |
| S8 | Direct-numeric probe (the LLM writes raw deltas) | Expected-negative control |

Targets for "sharpen", each with real headroom: **fabrication** (6.2%), **groundedness** (45.6%),
**abstention on retrieval_failure** (unmeasured, see §3), and **distractor robustness**
(unmeasured). Clean-condition correctness is the guardrail as well as a target.

**Part II: Install.** Edit θ_e4 toward QA. A4 (full SFT) = θ_qa, already available. A3 (LoRA from
e4) needs a cheap training run. Add **I6, task-vector scaling from e4**: how much of θ_qa's QA skill
can be recovered with a uniform α·τ, and how much more with per-module α? Plotting skill against
forgetting as α varies directly answers RQ5 (magnitude vs performance).

Hypotheses H1–H6 in Deliverable 3 §2 still stand. For Part I, H1 becomes: *targeted edits produce a
CI-positive reduction in fabrication and/or gain in abstention without lowering clean correctness
beyond its CI.*

## 6. Revised data splits

| Split | Source | Status |
|---|---|---|
| `edit-dev` | `qa_sft.jsonl` minus the 483 overlapping pairs. θ_qa trained on these, so its failures here are its "hard cases". | Available now |
| `iter-val` (rotated) | **Needs fresh QA pairs** from corpus chunks never used by `qa_sft`/`raft`/eval, made with the same teacher pipeline (`vendor/finetuning/prompts.py`). About 1–2k pairs, roughly $1 of Gemini at the family's observed $0.50 per 1k kept pairs. | **Open decision** (§8) |
| `final-test` | The frozen 2,000-item instrument. Scored at Gate 1 (baseline only, no edit is tuned against it) and at the end. | Available now |

Why not take `iter-val` from `qa_sft`? Because θ_qa trained on it, any score there is inflated and
would reward memorisation. Why not split the eval set in two? It would halve the final test and break
comparability with every published number.

## 7. Revised retention panel and Gate 1 target

**Retention** (forgetting is measured against θ_e4 *and* θ_qa, and both are reported):
- per-domain loss on e4's **native** held-out val bins (`data/retention/e4_native_val/`, SEC /
  case-law / web), the family's standard forgetting instrument;
- per-domain loss on v1's held-out bins (`data/retention/v1_heldout_val/`), for continuity with the
  pretraining reports;
- the Phase-0 copy and induction probes (`vendor/qa_probe/modal_phase0_probe.py`), ported to run
  locally on θ_e4 and θ_qa;
- clean-condition QA correctness, the guardrail for Part I.

Regression thresholds are **set at Gate 1 from measured noise floors**, not guessed. The Revision-1
values (SEC bpb +2%, copy ≥70%, CaseHOLD ≥0.46) were calibrated to the extended model and are
withdrawn.

**Gate 1 (concrete pass criterion).** Before any edit:
1. Download θ_e4 and θ_qa at pinned revisions, hash-lock them read-only under `checkpoints/_original/`.
2. A local deterministic harness that renders prompts exactly like `vendor/finetuning/eval.py`
   (custom `<|role|>` template, no BOS, greedy, 160 tokens) and computes token-F1, EM, fabrication
   and abstain with the vendor's metric code.
3. **Reproduce θ_qa clean: token-F1 0.229, EM 0.038, fabrication 0.062**, each inside the published
   CI. Also report the fraction of generations that match the reference outputs byte for byte, which
   measures GPU/library nondeterminism.
4. Measure all 4 conditions and the retention panel on θ_e4 and θ_qa, plus the §4 overlap split.
5. Write `GATE1_REPORT.md`, then stop and wait for a green flag.

## 8. Open decisions

1. **`iter-val` generation** (≈$1 of Gemini teacher calls; needs a Gemini key in `.env`): approve, or
   accept a weaker design without a rotated validation set.
2. **Editor LLM** for S5–S8: which model and API (affects cost and reproducibility logging).
3. **Compute:** the loop is sized for a local GPU (125M bf16 inference plus short LoRA runs). Modal
   is only needed if the local GPU isn't available.

---

## 🛑 REVISION 2 ENDS HERE

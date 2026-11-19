# wsp-study: weight-space programming on a 125M legal QA model

Can you install a new behaviour in a trained language model by editing its weights directly,
instead of fine-tuning it, without making it forget anything else?

This repo tests that on `slm-125m-sft` (θ_qa), a 125.8M LLaMA-style legal QA model.
- **The flaw:** θ_qa never says "I don't know". It abstains on 0 of 500 questions whose answer is
  not in the documents.
- **The target:** make it reply "not stated in the context" on those questions and nothing else.
- **The comparison:** LoRA, full fine-tuning, magnitude-matched random edits, AI-agent editors, and
  random search.

Every experiment was pre-registered before it ran (Part A of each report), and wrong predictions
are recorded as wrong. The 2,000-question final test was used only for the Gate 1 baseline and once
at the end.

**Start here:** [`docs/20_FINAL_REPORT.md`](docs/20_FINAL_REPORT.md). The full index is
[`docs/00_README.md`](docs/00_README.md).

## Headline results (frozen final test, paired against θ_qa)

| Arm | What changes | "I don't know" on missing answers | Clean F1 Δ | Forgetting (held-out) | Inside every guardrail |
|---|---|---|---|---|---|
| AND-gate edit | 1 neuron, 3 weight vectors | +0.724 | −0.018 [−0.030, −0.005] | +0.011% | no (clean F1, by 0.010) |
| LoRA lr 3e-4 | rank-16 adapters | +0.922 | +0.069 | +8.87% | no (forgetting) |
| Full fine-tune lr 1e-5 | every tensor | +0.932 | +0.068 | +5.86% | no (forgetting) |
| AI-agent editors A / B / C | gate + slice of a fine-tune delta | +0.91 / +0.87 / +0.87 | +0.00 / +0.06 / +0.01 | ≤ +0.43% | yes |
| Random search R1 | gate + slice of a fine-tune delta | +0.804 | +0.023 | +0.40% | yes |

**What the results show:**
- **The one-neuron edit mostly matches keywords.** On a pool where the answer sentence is deleted
  but the question's words remain, its discrimination falls from 0.72 to 0.12 (docs/14).
- **Random search matched the AI editors.** Random search over the same options also beat the
  hand-built edit (docs/17).
- **Self-recursion stopped at one step.** A second, recursively built neuron reached 87% held-out
  abstention but missed the clean-F1 guardrail by 0.002 (docs/18).

## Known limitations (being fixed)

- **The fine-tuning baselines** were not tuned to stay within the forgetting budget. Until they
  are, the "less forgetting than fine-tuning" comparison is not a fair one.
- **Random search** is a single run (R1).
- **False abstention** (refusing answerable questions) was not a guardrail. It rises in every arm,
  by +0.06 to +0.16 on the final test.
- **The "pure" edit is not purely hand-built.** Its output vector was found by a small gradient
  optimisation, and its detector was fitted by regression.
- **Scope:** one model, one behaviour, Claude-family editors, no independent replication.

## Layout

| Path | Contents |
|---|---|
| `docs/` | Proposal, pre-registrations and results (Part A and Part B of each gate), final report |
| `src/wsp/` | All study code: harness, edit engine, safety gate, every experiment |
| `wsp_runs/` | Result files only (`*_numbers.json`, tables, selections). Weights, deltas and per-item generations are not published (size). |
| `splits/` | Split manifests (rules, counts, hashes) |
| `vendor/` | Read-only copies of the code this study builds on (see `PROVENANCE.md`) |

**Not in the repo, because of size:**
- `data/`: the eval instrument, training data and retention sets.
- `checkpoints/`.

The models are on the Hugging Face Hub:
- [`Ace-2504/slm-125m-sft`](https://huggingface.co/Ace-2504/slm-125m-sft) (θ_qa);
- [`Ace-2504/slm-125m-e4`](https://huggingface.co/Ace-2504/slm-125m-e4).

## Ground rules followed

- The original checkpoints were hash-locked and never modified. Every arm is a separate delta.
- Every number has a bootstrap 95% CI.
- No editing decision used the final test set.

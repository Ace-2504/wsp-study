# v1-Extended QA-SFT — Plan (for review, nothing executed)

**Base model:** v1-extended — v1 (2.04B tok) + 515M continued-pretraining tokens
(75% fineweb-edu, 25% case-law, **0% SEC**). Checkpoint: Modal volume `slm-125m`,
`/data/cont/checkpoints/base/ckpt.pt` (720 MiB). Same tokenizer, same arch, same 1,024 ctx
as v1 — so the existing SFT tooling ports without modification.

**Teacher:** `gemini-3.1-flash-lite`.

> **Design constraint carried in from the previous QA-SFT study (v1→v2, 5 rounds):**
> that study produced no interpretable result — not because the model didn't improve, but
> because its measurement could not resolve whether it had. Every choice below is made to
> avoid repeating that. The specific failures are listed in §7.

---

## 1. What makes this base model worth fine-tuning

v1-extended is not just "a slightly different base." Phase 1 measured exactly what the
continuation did to each domain:

| Domain | Share of new data | Loss change vs v1 |
| --- | --- | --- |
| SEC | 0% | **+8.12% (worst)** |
| case-law | 25% | +1.00% |
| fineweb-edu | 75% | −1.68% (improved) |

Two facts follow, and they shape the whole plan:

1. **SEC is already damaged, and SEC was this model's only competitive advantage**
   (BPB 0.652 vs GPT-2's 0.768). SFT will damage it further. This model has an unusually
   clear thing to protect.
2. **The repo already proved the mitigation.** case-law at 25% representation forgot ~6×
   less than SEC at 0%. That is direct evidence that **replay works on this exact model** —
   it is not a hypothesis borrowed from literature.

So the interesting question here is *not* the one the previous study asked. It is:

> **Can we instruction-tune this model without re-damaging SEC — and does replay, which
> Phase 1 showed protects during pretraining, also protect during SFT?**

That is a real, falsifiable, publishable question with a strong prior from our own data.

---

## 2. What I am deliberately NOT doing

| Not doing | Why |
| --- | --- |
| A 10-round data-scaling curve (+1,000/round) | The previous study ran this to 4,000 pairs. No round was statistically separable from any other, or from the untrained base. Volume is not the lever; re-running it buys nothing. |
| Headlining a 1–5 LLM-judge score | On a model this weak, ~96% of answers score 1. It is a rare-event indicator wearing a 5-point costume. Kept as a *secondary* metric only. |
| Cumulative dataset with fixed epochs | Makes data volume and optimizer steps rise together, so forgetting cannot be attributed to either. |
| Headlining SFT-eval perplexity | Improved monotonically across 2 prior studies while quality did not move. It is uninformative on this setup — reported, never headlined. |

---

## 3. Phase 0 — the capability gate (run FIRST, ~$0.10, no training)

**Before generating a single QA pair**, establish what the base model can already do.

| Probe | Question | Why it gates everything |
| --- | --- | --- |
| **Verbatim copy** | Given a passage, reproduce one named sentence exactly | If it cannot copy, it cannot ground. Grounded SFT is then known-futile *in advance*, and that is a clean capacity result — not an ambiguous null after 10 rounds. |
| **Span selection** | Given a passage + question, is the answer span present in the output at all? | Separates "can locate" from "can phrase". |
| **Instruction-format response** | Does it emit a bounded answer, or continue the passage? | Base models continue text; measures the distance SFT must cover. |
| **SEC baseline** | Per-domain loss on v1's held-out val | The zero point for the forgetting curve. Must exist before training. |

**Decision rule, fixed now:**
- Copy rate ≥60% → proceed to full plan.
- Copy rate 20–60% → proceed, but pre-register that grounding gains will be capacity-capped.
- Copy rate <20% → **stop and report.** The ceiling is architectural; that is the finding,
  and it costs ~$0.10 instead of ~$15 and two weeks.

The previous study never ran this and consequently could not distinguish "task design
failed" from "model was never capable." One run settles it.

---

## 4. Dataset — QA pair counts

Generated **once**, frozen, then reused by every training arm. No cumulative growth.

| Split | Pairs | Purpose |
| --- | --- | --- |
| **Train** | **8,000** | Single fixed training set for all arms |
| **Eval — in-distribution** | **300** | Held out; source chunks quarantined |
| **Eval — out-of-distribution** | **200** | Non-corpus passages (news, Wikipedia, technical prose) |
| **Replay pool** | 2,000 | Raw pretraining text (SEC-weighted) for the replay arms |
| Raw generated | ~13,300 | @ 60% keep rate after filter/dedup/decontaminate |

**Why 8,000 train:** enough that the model is genuinely saturated, and past the point where
the prior study's curve had already gone flat. Train size is not the variable under study,
so it is set once at a comfortable value rather than swept.

**Why 300 + 200 eval, not 100:** *this is the single most important number in the plan.*
The previous study's 25-items-per-arm judge produced 95% intervals ~±0.2 wide on a metric
whose entire observed range was 0.3 — which is why nothing was resolvable. Eval size is
free (no GPU, no training) and it is what buys resolution. 300 items with a continuous
metric gives intervals roughly 5× tighter than what killed the last study.

**Composition (train):** 85% grounded/RAFT, 15% closed-book — carried over from v2, which
correctly identified that a 125M model cannot recall long-tail facts from single exposure.
Closed-book restricted to fineweb-edu (general knowledge only).
Tasks: qa 0.40 · summarize 0.25 · extract 0.20 · rewrite 0.15.
Domains: **v1-extended's own cumulative mix — SEC 33.6% · case-law 33.1% · fineweb-edu
33.3%** (derivation in §6.1). Applied to the QA pairs as well as the replay pool, so the
whole study follows one principle: *match the distribution the model actually learned.*
This replaces an earlier arbitrary "SEC-weighted to 50%". The mix is held **identical
across all six arms**, so it is a constant, not a confound — it can be changed later
without invalidating any within-study comparison.

---

## 5. Metrics — the actual fix

**Primary metrics are deterministic, continuous, and free.** No LLM judge in the headline.

| Metric | What it measures | Range |
| --- | --- | --- |
| **Copy rate** | % of grounded answers containing a verbatim ≥5-token span from the passage | 0–100%, continuous |
| **Fabrication rate** | % of answers containing a number **not** present in the passage | 0–100%, continuous |
| **Extractive F1** | token overlap of answer vs reference | 0–1, continuous |
| **Answer-span recall** | is the reference span present in the output | 0–100% |
| **False-abstention rate** | says "not stated" when the answer *is* present | 0–100% |

| Secondary | Note |
| --- | --- |
| Per-domain retention loss (sec / case-law / web) | **The forgetting curve.** Separately per domain — the SEC number is a headline result. |
| SFT-eval perplexity | Reported, explicitly labelled non-diagnostic. |

**DECIDED: no LLM judge.** All metrics in this study are deterministic and reproducible.
Consequence to state in the write-up: results are **not** numerically comparable to the
v1/v2 studies' judge scores. That is acceptable — those scores were shown to be
unresolvable, so continuity with them has little value. Every metric here has a defined
range, a defined denominator, and a bootstrap interval, and can be recomputed from saved
outputs at any time without an API call.

**Every number ships with a bootstrap 95% CI.** A result whose interval includes the
baseline is reported as "not resolved," not as a trend.

Fabrication rate deserves special mention: the previous study's clearest qualitative
failure — inventing "$3.5 billion to $8.6 billion" while the real figures sat unused in the
prompt — was described in prose for 5 rounds and never once counted. Here it is a primary
metric from round zero.

---

## 6. Training arms — 6 runs, one variable each

All arms: same 8,000-pair dataset, same seed, same eval. Initialized from v1-extended.

| # | Arm | Config | Question it answers |
| --- | --- | --- | --- |
| A | **Baseline (no SFT)** | — | True zero for every metric |
| B | **Standard SFT** | LR 3e-5, 3 epochs | Does SFT help at all, with error bars |
| C | **Low LR** | LR 1e-5, 3 epochs | Is forgetting LR-driven? |
| D | **High LR** | LR 1e-4, 3 epochs | " |
| E | **Fixed-step control** | LR 3e-5, 1 epoch | **Separates data volume from optimizer steps** — the confound the last study could not resolve |
| F | **Replay** | LR 3e-5, 3 epochs, +25% pretraining text at v1-extended's **own** cumulative mix (§6.1) | Does Phase 1's 6×-protection finding transfer from pretraining to SFT? |

### 6.1 Replay composition — v1-extended's own cumulative mix

Replay mirrors **what this model actually knows**, i.e. the composition of every token it
has ever trained on — v1's 2.04B plus the 520M continuation — not v1's corpus split and
not the continuation-only split.

| Domain | v1 (2,039M) | Continuation (520M) | **v1-extended cumulative (2,559M)** | Replay chunks (of 2,000) |
| --- | --- | --- | --- | --- |
| SEC | 42.2% | 0% | **33.6%** | 672 |
| case-law | 35.1% | 25.2% | **33.1%** | 661 |
| fineweb-edu | 22.8% | 74.8% | **33.3%** | 667 |

**Observation worth reporting on its own:** the continuation converted a legal-heavy model
into an almost perfectly balanced one (33.6 / 33.1 / 33.3). v1-extended is better described
as a general model with a legal background than as a legal specialist — which reframes what
"protecting its edge" even means.

**Why this beats the SEC-only alternative.** The obvious worry is that proportional replay
under-weights SEC and the protective effect vanishes into the error bars. The arithmetic
says otherwise: **Phase 1 measured ~6× protection at 25% representation** (case-law), and
33.6% is *above* that demonstrated threshold. The arm is therefore calibrated against a
measured effect on this exact model rather than an assumption that more is better. It is
also the composition anyone would actually ship.

*Not chosen: the continuation-only mix (0% SEC) — it would give zero protection to the one
domain already damaged, defeating the arm's purpose.*

**Optional 7th arm (~$0.19):** SEC-only replay, testing whether *targeting* the weak domain
beats *matching* the model's distribution. Not scheduled; decide after arm F reports.

---

**Arm F is the scientific payload.** Phase 1 established that 25% representation gave ~6×
less forgetting during continued pretraining. If that transfers to SFT, it is a genuinely
useful production result: *how to instruction-tune without losing your domain edge.*

Arms C/D/E answer the question the previous study left open — whether forgetting is driven
by optimisation rather than by what the tokens represent. The evidence pointed there and
was never tested.

---

## 7. Failures from the previous study, and where each is addressed

| # | Previous failure | Fix | § |
| --- | --- | --- | --- |
| 1 | Judge had no dynamic range (96% of answers scored 1) | Continuous automatic metrics as primary | §5 |
| 2 | n=25/arm → nothing resolvable; noise band borrowed from a different instrument | 300+200 eval items; bootstrap CIs on everything | §4, §5 |
| 3 | Per-item scores + `pair_id` discarded → no retrospective error bars | Persisted end-to-end from run 1 | §5 |
| 4 | Data volume confounded with optimizer steps | Arm E is the explicit control | §6 |
| 5 | Cross-study comparison invalid (decoding + judge both changed) | Instrument frozen before arm A; base scored on the identical instrument | §3, §8 |
| 6 | No OOD eval → couldn't tell reading from corpus-pattern-matching | 200-item OOD set, frozen at build time | §4 |
| 7 | Abstention trained but never measured | False-abstention is a primary metric | §5 |
| 8 | Capacity vs task-design never distinguished | Phase 0 copy probe, run before any spend | §3 |
| 9 | Replay was a vague "consider it past 15%" | Arm F is a pre-committed arm, not a mid-run decision | §6 |
| 10 | Stop rule fired below its own noise floor | Rules stated as CI exclusions (§8) | §8 |

---

## 8. Pre-registered outcomes (fixed now, before any run)

| Observation | Conclusion |
| --- | --- |
| Copy rate <20% at Phase 0 | Capacity ceiling. Stop, report. Total spend ~$0.10. |
| Arm B copy/F1 CI excludes arm A, upward | SFT works on this model. Proceed to full comparison. |
| Arm B CI includes arm A on all primary metrics | SFT does not measurably help at 125M — a **clean null with error bars**, which is the result the previous study wanted but could not support. |
| Arm F SEC retention CI excludes arm B, better | **Replay transfers to SFT.** Headline result. |
| Arm F ≈ arm B on SEC | Replay is pretraining-specific; report honestly against our own prior. |
| Arms C/D differ in forgetting but not quality | Forgetting is optimisation-driven — restates the last study's finding 2 as a general result. |
| Arm E ≈ arm B | Steps, not data volume, drive forgetting. |

---

## 9. Cost

**Modal** (L4 for SFT; measured from the prior study's runs — 4,000 pairs = 750 steps =
410s = $0.091):

| Item | Estimate |
| --- | --- |
| Phase 0 probes (inference only) | $0.10 |
| Arms B, C, D, F — 8,000 pairs × 3 epochs, ~1,500 steps ea. | 4 × $0.19 = $0.76 |
| Arm E — 1 epoch | $0.07 |
| Baseline + per-domain retention evals (6 × ~100s) | $0.15 |
| HF export of v1-extended (A100, one-off) | $0.15 |
| Contingency (reruns, failed jobs) | $1.00 |
| **Modal total** | **≈ $2.25** |

**Gemini teacher** (`gemini-3.1-flash-lite`, ~$0.50 per 1,000 kept pairs observed):

| Item | Estimate |
| --- | --- |
| 8,000 train pairs (~13,300 raw) | $4.00 |
| 500 eval pairs (in-dist + OOD) | $0.40 |
| Judge — 6 arms × 300 items | $0.60 |
| Regeneration contingency | $1.00 |
| **Teacher total** | **≈ $6.00** |

### **Total ≈ $8.25** (worst case ~$12)

Cost is not a constraint at this scale and must not drive any scientific decision. The real
cost is wall-clock: ~4h generation (rate-limited), ~1h training, ~2h analysis.

---

## 10. Sequence

1. Phase 0 capability gate → **STOP, report, await green flag**
2. Build + freeze datasets and eval sets; export v1-extended to HF format
3. Arm A baseline on the frozen instrument
4. Arms B–F (independent; can run in one batch)
5. Analysis with CIs → report
6. **STOP** — decide whether anything further is experiment or documentation

Hard stop after Phase 0, consistent with this repo's existing green-flag discipline.

---

## 11. Decisions taken

| # | Decision | Status |
| --- | --- | --- |
| 1 | **Scope: Phase 0 gate only**, then hard stop and report before any further spend | ✅ locked |
| 2 | **No LLM judge** — fully deterministic metrics (§5) | ✅ locked |
| 3 | **Publish v1-extended to the HF Hub** (clears the Phase 2a blocker). **No Space / no deployment.** Repo name confirmed with the user before any push. | ✅ locked |
| 4 | **Replay = v1-extended's own cumulative mix** (SEC 33.6 / case-law 33.1 / web 33.3), applied to both the QA pairs and the replay pool | ✅ locked |

Revised cost after dropping the judge: teacher ≈ **$5.40**, Modal ≈ **$2.25**,
**total ≈ $7.65**.

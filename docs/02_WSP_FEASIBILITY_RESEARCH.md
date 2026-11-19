# Deliverable 2 — Weight-Space Programming: Feasibility Research Report

**Question:** is WSP — *an external, more capable LLM that inspects a target model's weights/activations/
gradients, proposes targeted weight edits, applies them to a copy, evaluates, and iterates* — a
feasible way to improve legal QA in a 125M model, and how does it stand against established alternatives?

**Provenance / honesty note.** The literature below was gathered by a dedicated search pass over
Consensus and the web. Citations are reproduced with the verification flags from that pass: where an
arXiv ID or venue could **not** be independently re-verified, it is marked *(unverified)*. Nothing here
is fabricated; uncertain items are labelled uncertain rather than dropped or dressed up. Tiers:
**ESTABLISHED** = peer-reviewed + replicated · **EMERGING** = peer-reviewed but new/contested ·
**SPECULATIVE** = non-peer-reviewed or no empirical precedent.

> **Headline finding (read this first).** The *literal* WSP idea — an LLM reading raw weights and
> emitting the numeric edits in a closed loop — is **SPECULATIVE: it has no peer-reviewed precedent.**
> The only material that describes exactly this (the "Weight-Space Programming" posts reporting a
> "GPT-6 Astra edits 0.016% of Qwen3.5-0.8B, MATH-500 7.9%→48.1%" result) appears **only in blogs and
> social posts and references models that do not exist**; it is non-reproducible, the "editor" is a
> closed model, and in at least one write-up the weights were fed *as text in-prompt* with a Python
> kernel — i.e., not weight-space reasoning. **Treat it as hype, not evidence.** The nearest *real*
> research (SEAL) keeps an LLM in the loop but still makes the weight change by **gradient descent**.
> So the defensible WSP for this project is "LLM-as-controller over a gradient-based edit," and the
> project's value is to *measure that honestly against plain fine-tuning* — which is exactly what
> Deliverables 3–4 do.

---

## 1. The five technique families, and what each can and cannot do

### 1.1 Locate-then-edit knowledge editing — mature, but the wrong tool here

ROME (Meng et al., 2022, NeurIPS, arXiv:2202.05262), MEMIT (ICLR 2023, arXiv:2210.07229), MEND
(Mitchell et al., 2022, ICLR, arXiv:2110.11309), PMET (AAAI 2024, arXiv:2308.08742), WISE (NeurIPS
2024, arXiv:2405.14768), GRACE (NeurIPS 2023, arXiv:2211.11031).

- **What they do:** causal-trace a *specific factual association* to a few MLP weights and overwrite
  it ("Eiffel Tower is in → Rome"). They **edit a known fact; they do not install a procedure.**
- **Why they are the wrong tool for *this* target:** our target is a **question→answer-span mapping** —
  a *skill*, at 0/200 — not a fact to overwrite. This family was never built for that.
- **And they are documented to be dangerous at volume.** A strong, converging limitations literature:
  editing degrades general reasoning/QA/NLI (Gu et al., 2024, "Model Editing Harms General Abilities,"
  RECT); even a single edit can trigger collapse and nearly all methods collapse after a few
  sequential edits (Yang et al., 2024, "Butterfly Effect"/HardEdit); ripple-effect failures where
  in-context editing beats weight edits (Cohen et al., 2023, TACL, RippleEdits); degradation tracks
  the edited matrix's condition number / dominant singular directions (Ma et al., 2024, PRUNE; Zhang
  et al., 2026, REVIVE); even "safe" null-space editing (AlphaEdit) degrades competence at scale
  (Suresh et al., 2026, reproducibility study); a 2026 mechanistic result finds ROME/MEMIT *suppress*
  rather than genuinely rewrite (Holmov et al., 2026).

**Verdict:** ESTABLISHED for fact-overwriting, **near-hopeless for installing a QA skill**, and a
cautionary tale that *targeted edits break general ability unless tightly controlled* — which is why
Deliverable 3's retention panel and regression thresholds exist.

### 1.2 Task arithmetic & model merging — installs skills, but only ones already learned

Task vectors (Ilharco et al., 2023, ICLR, arXiv:2212.04089), tangent-space variant (Ortiz-Jiménez et
al., 2023), DARE (Yu et al., 2024, arXiv:2311.03099), evolutionary merging (Akiba et al., 2024,
arXiv:2403.13187, Sakana), Model Soups (Wortsman et al., 2022, ICML — *arXiv ID unverified*),
TIES-Merging (Yadav et al., NeurIPS 2023 — *ID unverified*).

- A task vector *is literally the delta from fine-tuning a model on the task.* Merging **transplants or
  blends behaviours that already exist** in some donor; evolutionary merging (CMA-ES over merge
  coefficients) can even surface emergent cross-skill ability.
- **Constraint:** there is no donor with legal-QA ability to transplant into our base — producing one
  means fine-tuning first. So merging is a *composition/packaging* tool, **not a shortcut around
  learning the skill.** Relevant later (compose a QA task-vector with SEC-preserving replay), not as
  the primary installer.

**Verdict:** ESTABLISHED; useful as a *composition* step once a QA skill exists; not a standalone path.

### 1.3 PEFT — the honest baseline, because it already *is* iterative weight editing

LoRA (Hu et al., 2021, arXiv:2106.09685), (IA)³ (Liu et al., 2022, NeurIPS, arXiv:2205.05638),
prefix/prompt/delta-tuning (Ding et al., 2023, *Nature Machine Intelligence*).

- These **install new behaviours via gradient descent on a copy of the weights**, are
  architecture-agnostic, routinely used at ≤125M, cheap, and mergeable back into the weights.
- **This is the crux of the whole project:** LoRA already does "modify a copy of the weights, evaluate,
  iterate" — using *gradients* instead of an LLM's guesses. Any WSP claim must be measured against it,
  or it is measuring against a straw man. (Deliverable 3, arm A3.)

**Verdict:** ESTABLISHED; the baseline WSP must beat or complement; the only proven way to *install* a
QA skill at this scale.

### 1.4 Activation steering / representation engineering — mostly inference-time, not weight edits

ITI (Li et al., 2023, NeurIPS), ActAdd (Turner et al., 2023), CAA (Rimsky et al., 2023), RepE (Zou et
al., 2023, arXiv:2310.01405).

- These **shift activations at inference**; they *elicit* latent behaviour, they do **not persist a
  missing skill in weights**. An "activation-informed" edit is only a *weight* edit if you bake the
  steering vector into parameters, which these methods generally do not.
- For WSP they are valuable as a **localization/diagnostic signal** (where does the question→span
  computation want to happen?), feeding the introspection stage — not as the installer.

**Verdict:** ESTABLISHED as inference-time control; **not weight-space skill installation**; useful as
introspection input.

### 1.5 LLM-as-optimizer / LLM-guided search — real, but optimizes *text and code, never weights*

OPRO (Yang et al., 2023, ICLR 2024, arXiv:2309.03409), FunSearch (Romera-Paredes et al., 2024,
*Nature*, DOI 10.1038/s41586-023-06924-6), Promptbreeder (Fernando et al., 2023), EvoPrompt (Guo et
al., *venue/year ambiguous*), LEO (Brahmachary et al., 2024).

- These are the real precedent for "LLM proposes solutions from a trajectory of (solution, score)
  pairs" — but the solutions are **prompts, programs, or low-dimensional design vectors. Never the
  weights of another network.**

**The precedent question, answered honestly:** a hard search for "LLM proposes numerical weight edits
to another network in a closed loop" returns **essentially nothing peer-reviewed.** The closest real
work is **SEAL — Self-Adapting LLMs (Zweiger et al., 2025, EMERGING):** an LLM generates its own
finetuning data and "self-edit" directives and an RL loop rewards edits by post-update performance —
but **the weight change is still a gradient-based SFT step, not LLM-authored numbers.** Hypernetworks
(MEND; Lo-Hp, Guan et al., 2025) emit weight updates, but they are *trained specialist nets*, not a
general LLM reasoning over raw tensors.

**Verdict:** the *orchestration* pattern is ESTABLISHED (OPRO/FunSearch); the *author-the-weights*
variant is SPECULATIVE with no reproducible precedent.

---

## 2. The dimensionality wall (why "LLM writes the weights" can't work blind)

Gradient-free optimization of raw weights is bounded by dimensionality. Evolution Strategies scale to
thousands of workers (Salimans et al., 2017, arXiv:1703.03864) **but on ~10⁴–10⁶-parameter policy
nets**; CMA-ES works for *merge coefficients* (hundreds of dims), not raw weights. Perturbation
variance grows with parameter count, so directly searching 125M weights needs thousands of full-model
evaluations per usable step. **Any gradient-free or LLM-authored edit is only viable over a tiny search
space** — merge weights, a LoRA's low-rank factors, a handful of steering coefficients. An LLM emitting
weight numbers is a *gradient-free optimizer over an enormous space*: the worst case of this wall. This
is the single hardest technical constraint and the reason Deliverable 3 confines any direct-numeric
probe (arm A8) to an expected-negative role.

---

## 3. Technical feasibility for *this* model

| Dimension | Assessment |
|---|---|
| **Engineering** | **Favourable.** 125.8M stock HF `LlamaForCausalLM`, loadable locally in fp32 (~0.5 GB), full `named_parameters`/hooks/autograd. The edit-eval loop runs on one machine. |
| **Introspection signals available** | Weights, activations (hooks), gradients (autograd), causal attribution (patching/logit-lens) — everything the editor could need, cheaply, at this scale. |
| **Scientific target** | **Hard.** QA is a *missing skill at 0%*, the regime where editing is weakest and PEFT is strongest. |
| **The literal WSP mechanism** | **Unproven.** No precedent for LLM-authored weight numbers; dimensionality wall; editing-breaks-ability literature. |
| **A defensible WSP mechanism** | **Feasible.** LLM-as-controller (select targets, generate/curate data, set hyperparameters, judge, analyse failures) wrapped around a gradient-based PEFT edit on a copy — SEAL-style, benchmarked against plain LoRA. |

---

## 4. Expected limitations (committed in advance)

1. **It will not install QA to anything like parity with a trained model**; expect a small, CI-positive
   movement at best (Deliverable 3, H1).
2. **Plain LoRA will likely match or beat the WSP arms on raw accuracy** — installing an input→output
   mapping is what gradient SFT is for (H3). WSP's defensible edge, if any, is *forgetting control and
   interpretability*, not accuracy.
3. **No new world knowledge, no general reasoning, no math, no closed-book facts** — out of reach for
   weight editing at 125M.
4. **The direct-numeric "LLM writes weights" variant is expected to fail** (≈ random edit); it is run
   only to *answer the question with data*.
5. **Editing can silently erode general ability** even when the targeted metric moves — hence the
   mandatory retention panel measured *against the original*.

---

## 5. Recommended methodology for the 125M model

The strongest version that rests on **established** technique, and the one Deliverables 3–4 implement:

1. **Reframe WSP as LLM-as-controller, not LLM-as-weight-author.** The external LLM does what LLMs
   demonstrably do well — analyse introspection evidence and eval failures, **select *where* to edit**
   (layers/modules/neurons/rank), generate and curate the extractive legal-QA data, set
   hyperparameters, and judge outputs. The actual weight delta is produced by a **grounded mechanism
   (gradient / bounded search) on a copy.** This is the SEAL pattern, de-hyped.
2. **Make LoRA-SFT the baseline, not the afterthought.** Report WSP only in relation to it.
3. **Keep the direct-numeric probe, small and honest**, so "can an LLM just write the weights?" is
   answered empirically rather than asserted.
4. **Confine any gradient-free search to a low-rank subspace** (LoRA factors / merge coefficients),
   never raw weights.
5. **Instrument forgetting to the editing-literature standard** — efficacy + generalization (paraphrase)
   + locality, *plus* a capability battery before/after, with perplexity as a cheap collapse surrogate
   and sequential-edit curves for the self-recursion study. Pre-register regression thresholds
   (Deliverable 3 §5).
6. **Localize with activation/attribution signals** (the one real use of the steering literature here)
   to choose edit sites, rather than editing blindly.

**In one line:** route *all* skill-installation through gradient-based PEFT on a copy; use the external
LLM for guidance, data, and evaluation — not for authoring weight numbers — and spend the project's
rigor on measuring, honestly, whether that guidance buys anything over plain LoRA.

---

## 6. References

Carried verbatim (with the same verification flags) from the search pass. *(unverified)* = arXiv
ID/venue not independently re-confirmed; **(Consensus)** = surfaced via the paper-search tool.

- Meng et al. *Locating and Editing Factual Associations in GPT* (ROME). NeurIPS 2022. arXiv:2202.05262.
- Meng et al. *Mass-Editing Memory in a Transformer* (MEMIT). ICLR 2023. arXiv:2210.07229.
- Mitchell et al. *Fast Model Editing at Scale* (MEND). ICLR 2022. arXiv:2110.11309.
- Li et al. *PMET: Precise Model Editing in a Transformer*. AAAI 2024. arXiv:2308.08742.
- Wang et al. *WISE: Rethinking the Knowledge Memory for Lifelong Model Editing*. NeurIPS 2024. arXiv:2405.14768.
- Hartvigsen et al. *Aging with GRACE*. NeurIPS 2023. arXiv:2211.11031.
- Dai et al. *Knowledge Neurons in Pretrained Transformers*. 2021. (Consensus).
- Cohen et al. *Evaluating the Ripple Effects of Knowledge Editing* (RippleEdits). TACL 2023. (Consensus).
- Gu et al. *Model Editing Harms General Abilities of LLMs* (RECT). 2024. (Consensus).
- Yang et al. *The Butterfly Effect of Model Editing* (HardEdit). 2024. (Consensus).
- Lin et al. *Navigating the Dual Facets: Evaluation of Sequential Memory Editing*. 2024. (Consensus).
- Ma et al. *Perturbation-Restrained Sequential Model Editing* (PRUNE). 2024. (Consensus).
- Zhang et al. *Spectral Characterization and Mitigation of Sequential Knowledge Editing Collapse* (REVIVE). 2026. (Consensus).
- Suresh et al. *Reproducibility Study of AlphaEdit*. 2026. (Consensus).
- Holmov et al. *One Mask to Rule Them All: On Hidden Facts after Editing*. 2026. (Consensus).
- Ilharco et al. *Editing Models with Task Arithmetic*. ICLR 2023. arXiv:2212.04089.
- Ortiz-Jiménez et al. *Task Arithmetic in the Tangent Space*. 2023. (Consensus).
- Yu et al. *Language Models are Super Mario* (DARE). 2024. arXiv:2311.03099.
- Akiba et al. *Evolutionary Optimization of Model Merging Recipes*. 2024. arXiv:2403.13187.
- Wortsman et al. *Model Soups*. ICML 2022. *(arXiv ID unverified.)*
- Yadav et al. *TIES-Merging*. NeurIPS 2023. *(arXiv ID unverified.)*
- Hu et al. *LoRA*. 2021. arXiv:2106.09685.
- Liu et al. *(IA)³ — Few-Shot PEFT Is Better and Cheaper than ICL*. NeurIPS 2022. arXiv:2205.05638.
- Ding et al. *Delta-tuning / parameter-efficient fine-tuning survey*. Nature Machine Intelligence 2023. (Consensus).
- Li et al. *Inference-Time Intervention* (ITI). NeurIPS 2023. (Consensus).
- Turner et al. *Steering Language Models With Activation Engineering* (ActAdd). 2023. (Consensus).
- Rimsky et al. *Steering Llama 2 via Contrastive Activation Addition* (CAA). 2023. (Consensus).
- Zou et al. *Representation Engineering*. 2023. arXiv:2310.01405.
- Yang et al. *Large Language Models as Optimizers* (OPRO). ICLR 2024. arXiv:2309.03409.
- Romera-Paredes et al. *Mathematical discoveries from program search with LLMs* (FunSearch). Nature 2024. DOI 10.1038/s41586-023-06924-6.
- Fernando et al. *Promptbreeder*. ICML 2024. *(web.)*
- Guo et al. *EvoPrompt*. *(venue/year ambiguous; web.)*
- Brahmachary et al. *LLM-Based Evolutionary Optimizer (LEO)*. 2024. (Consensus).
- Salimans et al. *Evolution Strategies as a Scalable Alternative to RL*. 2017. arXiv:1703.03864.
- Zweiger et al. *Self-Adapting Language Models* (SEAL). 2025. (Consensus).
- "Weight-Space Programming" — **non-peer-reviewed blog/social posts only**; references non-existent
  GPT-6 Astra / Qwen3.5-0.8B; SPECULATIVE, not a citable result.

---

## 🛑 DELIVERABLE 2 ENDS HERE

The feasibility verdict (defensible WSP = LLM-as-controller over gradient-based PEFT, benchmarked
against LoRA) is the premise of the experimental plan (Deliverable 3) and architecture (Deliverable 4).

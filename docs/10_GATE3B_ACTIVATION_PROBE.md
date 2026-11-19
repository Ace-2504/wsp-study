# Gate 3b-i — does θ_qa internally represent "the answer is not in the documents"?

**Approved 2026-11-10** (after the Gate 3a negative result): run the activation probe before any further
edit. **Part A below was written before any feature was extracted**; Part B is filled in afterwards.

## Part A — pre-registration

**Why.** Gate 3a showed that θ_qa's *output* carries no abstention preference (teacher-forced AUROC
0.42), and that no small scale or mask of the LoRA reaches valid abstention under the retention budget.
A small, surgical edit (e.g. rank-1: read a direction, write the abstain response) is only possible if
the information already exists *inside* the model. This probe tests that; it edits nothing.

**Models.**
- θ_qa (the target).
- θ_e4, the pre-SFT parent: was the representation there before QA fine-tuning?
- A **random-init** model with the same config (seed 1337). It is the control for what a probe can
  read from token identity alone, without learned processing.

**Features.**
- **Primary:** the residual-stream hidden state at the **last prompt token**, the position where
  generation starts, after `<|assistant|>\n`. Taken at every layer 0–12 (embeddings + 12 blocks).
- **Secondary:** the mean over all prompt tokens, at every layer.
- Prompts are rendered exactly as in evaluation: the custom template, the eval system prompt, no BOS.

**Data.** Pool v2, where document count is decorrelated from answerability, and every item is
labelled answer-absent (`abs*`) = 1 vs answerable (`ans*`) = 0.
- **Train:** `edit_dev` (12,732 items).
- **Choose:** `iter_val_0` (1,992 items). Used to pick the layer, the feature type and the
  regularisation.
- **Report:** `iter_val_1..3` pooled (6,756 items), never used for any choice.

**Probe.**
- L2 logistic regression on standardised features (train mean and std).
- λ ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1}, picked on `iter_val_0` for each layer.
- The layer is the one with the best `iter_val_0` pooled AUROC, for each model and feature type.

**Metrics.**
- AUROC on held-out data, **per document count k** (abs_k vs ans_k, so document count cannot drive
  it) and pooled.
- 95% CI by **cluster bootstrap over seeds**: the six items of one seed are resampled together;
  1,000 resamples, seed 0.

**Controls.**
1. **Shuffled labels:** the θ_qa probe retrained on permuted training labels, at the selected layer.
   It must give AUROC ≈ 0.5, which checks the pipeline.
2. **Lexical baseline:** a logistic regression on surface features with no model. The features are:
   - max and mean over documents of the IDF-weighted share of question words that appear in the
     document;
   - max share of question content words;
   - whether any document contains the question's rarest word;
   - question length and prompt length.

   IDF is computed on `edit_dev` documents only. This tests whether the probe reads more than
   word overlap.
3. **Random-init model.**
4. **θ_e4.**

**Decision rule** (θ_qa, the selected configuration, held-out):

| outcome | condition | consequence |
|---|---|---|
| **PRESENT** | AUROC ≥ 0.80 at every k | Design a representation-level edit (Gate 3b-ii, its own pre-registration). |
| **WEAK** | every k ≥ 0.65, some k < 0.80 | Report it. A read-and-write edit could at best reach this discrimination, so the recommended route becomes the retention-constrained LoRA. |
| **ABSENT** | any k < 0.65 | A small read-and-write edit is ruled out for this target at 125M. This is a negative result for the brief. |

**How the result is interpreted** (not part of the decision): the margin over the lexical baseline,
the random-init model and θ_e4 says *what kind* of representation it is. It could be surface word
overlap, something learned in pretraining, or something added by SFT. A probe that only matches the
lexical baseline would still be readable, but it would be a word-overlap detector, not reading
comprehension.

## Part B — results (`python -m wsp.act_probe`; raw: `wsp_runs/gate3b_probe/probe_results.json`, log `probe.log`)

**Verdict under the pre-registered rule: PRESENT.**
- θ_qa's last-prompt-token hidden state at **layer 5** separates answer-absent from answerable items
  at **AUROC 0.966 / 0.952 / 0.936** (k = 1 / 2 / 3 documents) on held-out folds 1–3 (6,756 items),
  with every 95% cluster-bootstrap CI above 0.92.
- **The lexical baseline is just as good**, so this is most likely a "do the documents contain the
  question's words?" signal, not evidence of reading comprehension. It is still a readable direction
  a small edit can use (see the interpretation below).

Held-out AUROC (95% CI, cluster bootstrap over seeds):

| features | selected layer (on iter_val_0) | k = 1 | k = 2 | k = 3 | pooled |
|---|---|---|---|---|---|
| **θ_qa, last token** | 5 | **0.966** [0.959, 0.972] | **0.952** [0.944, 0.959] | **0.936** [0.928, 0.946] | 0.952 |
| θ_qa, prompt mean | 5 | 0.832 | 0.764 | 0.715 | 0.775 |
| θ_e4, last token | 8 | 0.834 | 0.805 | 0.758 | 0.799 |
| θ_e4, prompt mean | 8 | 0.790 | 0.727 | 0.680 | 0.737 |
| random init, last token | 1 | 0.727 | 0.652 | 0.635 | 0.673 |
| random init, prompt mean | 0 | 0.743 | 0.680 | 0.654 | 0.698 |
| **lexical baseline (no model)** | — | **0.963** [0.955, 0.970] | **0.968** [0.962, 0.974] | **0.968** [0.962, 0.974] | 0.966 |
| shuffled labels (θ_qa, last, layer 5) | — | 0.470 | 0.463 | 0.486 | 0.474 |

How the θ_qa last-token probe develops by layer (dev AUROC, iter_val_0):

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev AUROC | 0.50 | 0.58 | 0.65 | 0.77 | 0.75 | **0.94** | 0.93 | 0.93 | 0.92 | 0.91 | 0.90 | 0.88 | 0.84 |

(Layer 0 is exactly 0.50 because the last prompt token is always the same token, so its embedding
carries no information. That is a built-in sanity check.)

### Interpretation (pre-registered as interpretation, not decision)

1. **The pipeline is sound.**
   - Shuffled labels give 0.47–0.49.
   - Layer 0 gives exactly 0.50.
   - The θ_qa signal appears abruptly between layers 4 and 5, and then fades towards the output
     (0.84 at layer 12).
2. **The information is inside θ_qa but does not reach the output.**
   - Layer 5 reads absent-vs-answerable at 0.95.
   - The output preference for the abstain string is 0.42, the wrong way (Gate 3a §4).
   - θ_qa *has* the signal and does nothing with it. This is the situation where a small read-and-write
     edit is plausible: route an existing direction to an existing output. Learning a new feature is
     not needed.
3. **SFT built most of it.** θ_e4 reaches 0.80 and the random-init model 0.67. QA fine-tuning took a
   weaker pretrained signal and made it sharp and early (layer 5 vs 8).
4. **It is most likely word overlap, not comprehension.**
   - The model-free lexical baseline (how much of the question's IDF-weighted vocabulary appears in the
     best-matching document) scores **0.966**, equal to or above the θ_qa probe at every k.
   - On this pool, "answer present" is close to "some document shares the question's rare words". That
     follows from how the pool was built: questions were written from the golden passage, so it
     overlaps the question more than the hardest distractor does.
   - The probe has not been shown to carry anything beyond overlap. Testing that would need a pool
     whose distractors are matched on overlap, which this one is not.
5. **What this means for the target.**
   - An edit built on this direction would implement roughly "abstain when no document shares the
     question's vocabulary". That is useful and legitimate abstention behaviour.
   - Its limit is the lexical one: it would miss distractors that share the words but not the answer.
   - The Gate 2 LoRA may have learned the same rule. Neither has been tested on overlap-matched
     distractors.

### Consequence (per the decision rule)

PRESENT → **Gate 3b-ii: a representation-level edit**, with its own pre-registration before it runs:
- **read:** the layer-5 probe direction;
- **write:** the abstain response, via a rank-1 (or very low-rank) weight update in a later layer;
- **checks:** the same dev-fold selection, held-out folds, matched random controls and full retention
  as Gates 2–3a;
- **the bar:** the Gate 3a frontier (no valid abstention inside +0.5% retention) and the Gate 2 LoRA's
  trade-off.

The independent editor-agent arm (step 2 of the remaining plan) gets this read-and-write edit as its
main tool.

Also recommended, and cheap: an **overlap-matched control pool**, with distractors chosen to match the
golden passage's question overlap. It would show whether any abstention we install is more than a
word-overlap rule. It applies to the LoRA as well.

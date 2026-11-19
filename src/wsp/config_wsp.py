"""Single source of truth for every WSP threshold. Changing a value here after an arm has run
invalidates that arm's pre-registration — log the change in docs and re-run.

Two kinds of limits:
  * GUARDRAILS — approved by Harman at the Gate 1 green flag (docs/06_GATE1_REPORT.md §9).
    A breach fails the arm regardless of its target gain.
  * SAFETY — engineering limits from Deliverable 4 that catch broken edits BEFORE the (slower)
    evaluation runs: non-finite weights, runaway magnitudes, exploding activations.
"""
from __future__ import annotations

# ---- GUARDRAILS (approved 2026-11-10) ----
RETENTION_MAX_REL_INCREASE = 0.005      # held-out loss, any domain, vs θ_qa: <= +0.5%
CLEAN_F1_MIN_DELTA_LOWER_CI = -0.02     # paired Δ clean token-F1 vs θ_qa: lower CI bound >= -0.02
VERBATIM_COPY_MIN = 0.93                # needle verbatim-cue copy rate (θ_qa lower CI bound)
FABRICATION_MUST_NOT_RISE = True        # paired Δ clean fabrication vs θ_qa must not be significantly > 0

# ---- SAFETY (engineering; checked before evaluation) ----
MAX_REL_DELTA_PER_TENSOR = 0.25         # ‖ΔW‖/‖W‖ for any single tensor (τ's max is 0.052)
MAX_REL_DELTA_GLOBAL = 0.05             # ‖Δθ‖/‖θ‖ over all tensors (τ is 0.0167)
MAX_ACTIVATION_RATIO = 2.0              # max |hidden| per layer on canary batch, vs θ_qa
MAX_CANARY_LOSS_REL_INCREASE = 0.10     # quick 64-window loss check, catches catastrophic edits early

# ---- loop evaluation protocol ----
RETENTION_FAST_WINDOWS = 512            # first N windows per e4_native domain (validated vs full set)
CANARY_WINDOWS = 64
GEN_BATCH = 32
ABS_KINDS = ("abs1", "abs2", "abs3")     # primary target: abstention when the answer is absent (1/2/3 docs)
ANS_KINDS = ("ans1", "ans2", "ans3")     # guardrails: answerable; ans1 = the "clean" condition
CLEAN_KIND = "ans1"
MIN_DISCRIMINATION_REPORTED = True      # abstain(abs_k) - abstain(ans_k) reported for every k (shortcut check)
SEED = 1337

"""Weight-Space Programming experiment code for the slm-125m family.

Gate 1 modules (no editing yet):
    paths        - every on-disk location, in one place
    checkpoints  - hash-locked originals: verify + load (never written)
    metrics      - QA metrics, byte-faithful to vendor/finetuning/eval.py, plus bootstrap
    qa_eval      - deterministic grounded-QA harness over the frozen 2,000-item instrument
    retention    - per-domain held-out loss (forgetting panel)
    probes       - Phase-0 copy / induction probes, ported to run locally
    overlap      - train/eval 13-gram overlap + the edit-dev quarantine list
"""

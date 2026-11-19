"""Every path the experiment touches. Relative to the project root, never to the CWD."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
EVAL_MESSAGES = DATA / "eval" / "eval_messages.jsonl"   # SHA 16e12b23… == reference manifests
EVAL_SHA256 = "16e12b23bab88f42ac35234e31dbcdd63f9e1037df5c3f0e3d462d9808896d48"
QA_SFT = DATA / "train" / "qa_sft.jsonl"                 # θ_qa's training data; few-shot source
RAFT = DATA / "train" / "raft.jsonl"
REFERENCE = DATA / "reference_results"
RETENTION = {
    "e4_native": DATA / "retention" / "e4_native_val",   # held-out val of e4's 2.5B corpus
    "v1_heldout": DATA / "retention" / "v1_heldout_val", # v1's held-out val (pretraining reports)
}

CKPT_ORIGINAL = ROOT / "checkpoints" / "_original"
LOCK = CKPT_ORIGINAL / "LOCK.json"
ORIGINALS = {"e4": CKPT_ORIGINAL / "e4", "qa": CKPT_ORIGINAL / "qa"}

RUNS = ROOT / "wsp_runs"
SPLITS = ROOT / "splits"

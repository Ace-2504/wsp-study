"""Hash-locked original checkpoints. This module can verify and load them; it never writes them.

The originals live under checkpoints/_original/{e4,qa}, are marked read-only on disk, and every
file's SHA-256 is recorded in LOCK.json. `verify_lock()` must pass before any run that reports a
number, so a silently modified original can never contaminate a result.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import paths


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_lock(tags=("e4", "qa")) -> dict:
    """Recompute every locked file's hash. Raises on any mismatch or missing file."""
    lock = json.loads(paths.LOCK.read_text())
    for tag in tags:
        for fname, want in lock[tag]["files"].items():
            got = _sha256(paths.ORIGINALS[tag] / fname)
            if got != want:
                raise RuntimeError(f"ORIGINAL MODIFIED: {tag}/{fname} sha {got[:12]} != lock {want[:12]}")
    return {t: lock[t]["source"] for t in tags}


def model_dir(tag_or_path: str) -> Path:
    return paths.ORIGINALS[tag_or_path] if tag_or_path in paths.ORIGINALS else Path(tag_or_path)


def load(tag_or_path: str, device: str = "cuda", dtype: str = "bfloat16"):
    """Load (model, tokenizer, eos_id) exactly as vendor/finetuning/eval.py does for the custom family:
    bf16 weights, eval mode, config.eos_token_id forced to <|eos|>."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    d = model_dir(tag_or_path)
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForCausalLM.from_pretrained(d, torch_dtype=getattr(torch, dtype)).to(device).eval()
    eos = tok.convert_tokens_to_ids("<|eos|>")
    model.config.eos_token_id = eos
    return model, tok, eos

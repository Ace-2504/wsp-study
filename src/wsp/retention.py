"""Forgetting panel: per-domain held-out next-token loss, per window, with bootstrap CIs.

Matches vendor/pretraining/train_core.evaluate_loss numerically (bf16 autocast, labels=input_ids,
mean over each 1024-token window; the vendor's per-batch mean of equal-size batches equals the
mean of per-window means). Per-window losses are saved, so any subset protocol (e.g. Phase-0 "D":
first 40 batches of 8 = first 320 windows) is derivable without re-running, and a fast loop-time
subset can be validated against the full set.

Sets: e4_native = e4's own held-out val (sec / case-law / fineweb-edu).
      v1_heldout = v1's held-out val, case-law + fineweb-edu only, made STRICTLY DISJOINT from
      e4_native (verified at Gate 1: v1's SEC bins are byte-identical to e4_native's, and 760 of
      v1's 4,580 fineweb-edu windows also occur in e4_native; those are dropped here so the two
      sets never double-count the same text).

    python -m wsp.retention --model qa --name qa
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import checkpoints, metrics, paths

SEQ = 1024
SETS = {"e4_native": ("sec", "case-law", "fineweb-edu"), "v1_heldout": ("case-law", "fineweb-edu")}


def _raw(set_name: str, source: str) -> np.ndarray:
    files = sorted(paths.RETENTION[set_name].glob(f"{source}-*.bin"))
    return np.concatenate([np.memmap(f, dtype=np.uint16, mode="r").reshape(-1, SEQ) for f in files])


def windows(set_name: str, source: str) -> np.ndarray:
    w = _raw(set_name, source)
    if set_name == "v1_heldout":
        seen = {r.tobytes() for r in _raw("e4_native", source)}
        w = w[np.array([r.tobytes() not in seen for r in w], dtype=bool)]
    return w


def per_window_loss(model, ids: np.ndarray, batch: int = 16, device: str = "cuda") -> np.ndarray:
    import torch
    import torch.nn.functional as F
    out = np.empty(len(ids), np.float64)
    with torch.no_grad():
        for s in range(0, len(ids), batch):
            x = torch.from_numpy(ids[s:s + batch].astype(np.int64)).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids=x).logits
            l = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                                x[:, 1:].reshape(-1), reduction="none").view(x.size(0), -1)
            out[s:s + x.size(0)] = l.mean(1).double().cpu().numpy()
    return out


def evaluate(model, out_dir: Path, name: str, max_windows: int = 0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for set_name, sources in SETS.items():
        for src in sources:
            ids = windows(set_name, src)
            if max_windows:
                ids = ids[:max_windows]
            t0 = time.time()
            l = per_window_loss(model, ids)
            np.save(out_dir / f"{name}__{set_name}__{src}.npy", l)
            m, lo, hi = metrics.bootstrap_ci(l)
            summary[f"{set_name}/{src}"] = {
                "n_windows": int(len(l)), "loss": m, "ci": [lo, hi], "ppl": float(np.exp(m)),
                "phase0_D_protocol_loss": float(l[:320].mean()), "secs": round(time.time() - t0, 1)}
            print(f"  [{name}] {set_name}/{src}: n={len(l)} loss {m:.4f} [{lo:.4f},{hi:.4f}] "
                  f"ppl {np.exp(m):.2f}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--out-dir", default=str(paths.RUNS / "gate1" / "retention"))
    a = ap.parse_args()
    checkpoints.verify_lock()
    model, _, _ = checkpoints.load(a.model)
    out = Path(a.out_dir)
    s = evaluate(model, out, a.name, a.max_windows)
    (out / f"{a.name}.json").write_text(json.dumps({"model": a.model, "summary": s}, indent=1))


if __name__ == "__main__":
    main()

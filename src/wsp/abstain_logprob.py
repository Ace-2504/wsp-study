"""Gate 3a POST-HOC diagnostic (exploratory, not pre-registered; dev fold iter_val_0 only).

Greedy abstention jumps from ~0.01 to ~0.3 between 30% and 50% of the LoRA delta. Is the underlying
signal also absent at small scale, or is it present but below the greedy decision threshold?

Teacher-forced score per item: S = log P("not stated in the context<|eos|>" | prompt), summed over
the target tokens. For each scale of Δ_lr3e-4 we report
  * AUROC of S separating absent (abs_k) from answerable (ans_k) items, per document count k
    (k-matched, so document count cannot drive it), and pooled
  * mean S on absent and on answerable items.
AUROC 0.5 = no signal; 1.0 = perfect separation, regardless of whether greedy decoding abstains.

    python -m wsp.abstain_logprob
"""
from __future__ import annotations

import json

import numpy as np

from . import checkpoints, engine, paths, splits
from .qa_eval import render

OUT = paths.RUNS / "gate3a"
TARGET = "not stated in the context"
SCALES = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)


def auroc(pos, neg) -> float:
    """P(score_pos > score_neg) with ties as 1/2 (Mann-Whitney)."""
    x = np.concatenate([pos, neg]); r = np.empty(len(x))
    o = np.argsort(x, kind="mergesort"); r[o] = np.arange(1, len(x) + 1)
    for v in np.unique(x):                                  # average ranks for ties
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def scores(model, tok, items, batch=16) -> np.ndarray:
    import torch
    tgt = tok(TARGET, add_special_tokens=False).input_ids + [1]          # + <|eos|>
    seqs = []
    for it in items:
        p = tok(render(it["messages"][0]["content"], it["messages"][1]["content"], []), add_special_tokens=False).input_ids
        p = p[-(1024 - len(tgt)):]
        seqs.append((p, tgt))
    out = np.zeros(len(seqs))
    with torch.no_grad():
        for s in range(0, len(seqs), batch):
            b = seqs[s:s + batch]
            L = max(len(p) + len(t) for p, t in b)
            ids = torch.zeros((len(b), L), dtype=torch.long); att = torch.zeros((len(b), L), dtype=torch.long)
            for j, (p, t) in enumerate(b):
                ids[j, :len(p) + len(t)] = torch.tensor(p + t); att[j, :len(p) + len(t)] = 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits.float()
            lp = torch.log_softmax(lg, -1)
            for j, (p, t) in enumerate(b):
                pos = torch.arange(len(p) - 1, len(p) + len(t) - 1)
                out[s + j] = float(lp[j, pos, torch.tensor(t)].sum())
    return out


def main():
    checkpoints.verify_lock()
    _, tok, _ = checkpoints.load("qa")
    base = engine.load_state("qa")
    d = engine.load_delta(paths.RUNS / "gate2" / "deltas" / "lora_lr0.0003.safetensors")
    items = splits.load("iter_val_0", kinds=["ans1", "ans2", "ans3", "abs1", "abs2", "abs3"])
    kind = np.array([i["kind"] for i in items])
    res = {}
    for s in SCALES:
        m = engine.materialize(base, d, scale=s)
        S = scores(m, tok, items)
        r = {"auroc_k": {k: auroc(S[kind == f"abs{k}"], S[kind == f"ans{k}"]) for k in (1, 2, 3)},
             "auroc_pooled": auroc(S[np.char.startswith(kind, "abs")], S[np.char.startswith(kind, "ans")]),
             "mean_logp_absent": float(S[np.char.startswith(kind, "abs")].mean()),
             "mean_logp_answerable": float(S[np.char.startswith(kind, "ans")].mean())}
        res[str(s)] = r
        print(f"scale {s:<5} AUROC k=1/2/3 " + "/".join(f"{r['auroc_k'][k]:.3f}" for k in (1, 2, 3)) +
              f"  pooled {r['auroc_pooled']:.3f}  mean logP absent {r['mean_logp_absent']:.2f} answerable {r['mean_logp_answerable']:.2f}", flush=True)
        del m
    (OUT / "diagnostic_abstain_logprob.json").write_text(json.dumps(
        {"note": "POST-HOC exploratory diagnostic, dev fold iter_val_0 only; not pre-registered", "target": TARGET,
         "delta": "gate2 lora_lr0.0003", "n_items": len(items), "results": res}, indent=1))


if __name__ == "__main__":
    main()

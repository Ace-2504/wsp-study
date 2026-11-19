"""QA metrics.

The scoring functions are copied VERBATIM from vendor/finetuning/eval.py (that file imports
`modal` at module load, so it cannot be imported here). `check_against_reference()` proves the copy
is faithful by re-scoring the stored reference generations and requiring bit-identical scores.
"""
from __future__ import annotations

import re

import numpy as np

ABSTAIN = "not stated in the context"


# ---------------- verbatim from vendor/finetuning/eval.py ----------------
def _words(s):
    return [w for w in "".join(c if c.isalnum() or c.isspace() else " " for c in s.lower()).split()]

def token_f1(pred, ref):
    p, r = _words(pred), _words(ref)
    if not p or not r:
        return 0.0
    common = {}
    for w in p:
        common[w] = common.get(w, 0) + 1
    overlap = sum(min(cnt, r.count(w)) for w, cnt in common.items())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(r)
    return 2 * prec * rec / (prec + rec)

def exact_match(pred, ref):
    n = lambda s: " ".join(_words(s))
    return float(n(pred) == n(ref))

def _norm_num(s):
    s = s.replace(",", "").rstrip("%").rstrip(".")
    return s

def fabrication(pred, ctx):
    ctx_norm = _norm_num(ctx.replace(",", ""))
    nums_p = {_norm_num(x) for x in re.findall(r"\d[\d,\.]*", pred)}
    nums_c = {_norm_num(x) for x in re.findall(r"\d[\d,\.]*", ctx)}
    bad = [x for x in nums_p if x and x not in nums_c and x not in ctx_norm]
    return float(len(bad) > 0)

def is_abstain(pred):
    return float(ABSTAIN in pred.lower())

def false_abstain(pred, answerable):
    return float(answerable and (ABSTAIN in pred.lower()))
# ---------------- end verbatim ----------------


def score_item(resp: str, ref: str, user: str, answerable: bool) -> dict:
    return {"token_f1": token_f1(resp, ref), "exact_match": exact_match(resp, ref),
            "fabrication": fabrication(resp, user), "abstain": is_abstain(resp),
            "false_abstain": false_abstain(resp, answerable),
            "resp_len_words": float(len(resp.split()))}


def bootstrap_ci(vals, n: int = 5000, seed: int = 0):
    """Identical to the vendor's bootstrap (same n, same seed) so CIs are comparable."""
    a = np.asarray(list(vals), float)
    if a.size == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    ms = rng.choice(a, (n, a.size), replace=True).mean(1)
    return (float(a.mean()), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5)))


def paired_delta_ci(a: dict, b: dict, n: int = 5000, seed: int = 0) -> dict:
    """CI on mean(A - B) over shared keys, one shared resample — the correct model-vs-model test."""
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys], float)
    if d.size == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    ms = d[rng.integers(0, d.size, (n, d.size))].mean(1)
    lo, hi = float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))
    return {"n": int(d.size), "delta": float(d.mean()), "ci": [lo, hi],
            "significant": bool(lo > 0 or hi < 0)}


def aggregate(items: list[dict]) -> dict:
    """Per-condition aggregate in the vendor's result schema."""
    out = {}
    for cond in sorted({it["cond"] for it in items}):
        rows = [it["scores"] for it in items if it["cond"] == cond]
        out[cond] = {"n": len(rows),
                     "token_f1": bootstrap_ci(r["token_f1"] for r in rows),
                     "exact_match": bootstrap_ci(r["exact_match"] for r in rows),
                     "fabrication": bootstrap_ci(r["fabrication"] for r in rows),
                     "abstain_rate": bootstrap_ci(r["abstain"] for r in rows),
                     "false_abstain_rate": bootstrap_ci(r["false_abstain"] for r in rows)}
    return out


def check_against_reference(per_item: list[dict]) -> dict:
    """Re-score stored reference generations with THIS module; count any score that differs."""
    mism = 0
    for it in per_item:
        mine = score_item(it["resp"], it["ref"], it["user"], it["answerable"])
        for k in ("token_f1", "exact_match", "fabrication", "abstain", "false_abstain"):
            if mine[k] != it["scores"][k]:
                mism += 1
    return {"n_items": len(per_item), "score_mismatches": mism}

"""The loop-time evaluator: one call scores a model on the target, the guardrails and retention.

    rep = evaluate(model, tok, split="iter_val_0")       # dict of aggregates + per-item results
    verdict = guardrails(rep, baseline_rep)              # approved Gate 1 thresholds, paired vs θ_qa

Target   : abstention rate on abs1/abs2/abs3 items (answer absent; 1/2/3 documents).
Guardrail: ans1 ("clean") token-F1 + fabrication (approved); ans1..3 F1 / false abstention reported.
Shortcut : discrimination abstain(abs_k) - abstain(ans_k) for every document count k.
Retention: first RETENTION_FAST_WINDOWS windows of each e4_native domain (validated against the
           full held-out set at Gate 2), loss vs θ_qa.
All generation is batched greedy (wsp.batchgen) — the same procedure for every arm.
"""
from __future__ import annotations

import numpy as np

from . import batchgen, config_wsp as C, metrics, splits
from .retention import per_window_loss, windows


def retention_ids(n: int = C.RETENTION_FAST_WINDOWS) -> dict:
    return {src: windows("e4_native", src)[:n] for src in ("sec", "case-law", "fineweb-edu")}


def evaluate(model, tok, split: str, kinds=C.ANS_KINDS + C.ABS_KINDS,
             ret_ids: dict | None = None, eos: int = 1) -> dict:
    items = splits.load(split, kinds=list(kinds))
    resp = batchgen.generate(model, tok, eos, [(i["messages"][0]["content"], i["messages"][1]["content"])
                                               for i in items], C.GEN_BATCH)
    per = []
    for it, r in zip(items, resp):
        ref, user = it["messages"][2]["content"], it["messages"][1]["content"]
        answerable = it["kind"].startswith("ans")
        per.append({"id": it["id"], "kind": it["kind"], "source": it["source"], "resp": r,
                    "scores": metrics.score_item(r, ref, user, answerable)})
    agg = {}
    for k in kinds:
        rows = [p["scores"] for p in per if p["kind"] == k]
        agg[k] = {m: metrics.bootstrap_ci(x[m] for x in rows)
                  for m in ("token_f1", "exact_match", "fabrication", "abstain", "false_abstain")}
        agg[k]["n"] = len(rows)
    out = {"split": split, "agg": agg, "per_item": per}
    if ret_ids is not None:
        out["retention"] = {src: per_window_loss(model, ids).tolist() for src, ids in ret_ids.items()}
    return out


def _by(rep, kinds, m):
    kinds = (kinds,) if isinstance(kinds, str) else kinds
    return {p["id"]: p["scores"][m] for p in rep["per_item"] if p["kind"] in kinds}


def discrimination(rep: dict) -> dict:
    """abstain(absent, k docs) - abstain(answerable, k docs) for k = 1, 2, 3. A real abstention skill is
    high at EVERY k; a document-count shortcut is high at one k and ~0 (or negative) at the others."""
    out = {}
    for k in (1, 2, 3):
        a = [p["scores"]["abstain"] for p in rep["per_item"] if p["kind"] == f"abs{k}"]
        b = [p["scores"]["abstain"] for p in rep["per_item"] if p["kind"] == f"ans{k}"]
        out[k] = float(np.mean(a) - np.mean(b)) if a and b else None
    vals = [v for v in out.values() if v is not None]
    out["min"] = min(vals) if vals else None
    return out


def compare(rep: dict, base: dict) -> dict:
    """Paired deltas vs the baseline report (same items), and the approved guardrail verdict."""
    d = {"target_abstain": metrics.paired_delta_ci(_by(rep, C.ABS_KINDS, "abstain"), _by(base, C.ABS_KINDS, "abstain"))}
    for k in C.ABS_KINDS:
        d[f"{k}.abstain"] = metrics.paired_delta_ci(_by(rep, k, "abstain"), _by(base, k, "abstain"))
    for k in C.ANS_KINDS + ("ans_all",):
        kk = C.ANS_KINDS if k == "ans_all" else k
        for m in ("token_f1", "fabrication", "false_abstain"):
            d[f"{k}.{m}"] = metrics.paired_delta_ci(_by(rep, kk, m), _by(base, kk, m))
    d["discrimination"] = discrimination(rep)
    c = C.CLEAN_KIND
    g = {"clean_f1": d[f"{c}.token_f1"]["ci"][0] >= C.CLEAN_F1_MIN_DELTA_LOWER_CI,
         "fabrication": not (d[f"{c}.fabrication"]["significant"] and d[f"{c}.fabrication"]["delta"] > 0)}
    if "retention" in rep and "retention" in base:
        r = {}
        for src in rep["retention"]:
            a, b = np.array(rep["retention"][src]), np.array(base["retention"][src])
            r[src] = {"rel": float(a.mean() / b.mean() - 1),
                      "delta": metrics.paired_delta_ci(dict(enumerate(a)), dict(enumerate(b)))}
        d["retention"] = r
        g["retention"] = all(v["rel"] <= C.RETENTION_MAX_REL_INCREASE for v in r.values())
    d["guardrails"] = g
    d["guardrails_pass"] = all(g.values())
    return d

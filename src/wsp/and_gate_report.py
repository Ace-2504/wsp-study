"""Gate 3b-iv report numbers: held-out folds 1-3 POOLED, paired vs θ_qa (as pre-registered in docs/13),
plus full retention vs θ_qa and selected-vs-control paired abstention / discrimination differences.

    python -m wsp.and_gate_report
"""
from __future__ import annotations

import json

import numpy as np

from . import config_wsp as C, metrics, paths
from .loop_eval import _by, compare, discrimination

OUT = paths.RUNS / "gate3b_andgate"
G2 = paths.RUNS / "gate2"
ARMS = ("selected", "ctrl_shuffled_read", "ctrl_write_only", "ctrl_ablation", "random_s0", "random_s1", "random_s2")


def pooled(name):
    per = []
    for f in (1, 2, 3):
        p = OUT / "reports" / f"{name}__iter_val_{f}.json" if name != "qa" else G2 / f"baseline_qa_iter_val_{f}.json"
        if not p.exists():
            return None
        per += json.loads(p.read_text())["per_item"]
    return {"per_item": per}


def disc_items(rep):
    """Per-item contribution so discrimination differences can be paired: abs items +abstain, ans items −abstain, per k."""
    return rep


def main():
    base = pooled("qa")
    out = {"n_items": {k: sum(p["kind"] == k for p in base["per_item"]) for k in C.ANS_KINDS + C.ABS_KINDS}, "heldout": {}}
    reps = {}
    for a in ARMS:
        r = pooled(a)
        if r is None:
            out["heldout"][a] = "missing (safety veto)"
            continue
        reps[a] = r
        c = compare(r, base)
        out["heldout"][a] = {k: c[k] for k in ("target_abstain", "ans1.token_f1", "ans_all.token_f1", "ans_all.false_abstain",
                                               "ans1.fabrication", "discrimination", "guardrails", "guardrails_pass")}
        out["heldout"][a]["validity"] = c["discrimination"]["min"] >= 0.5 * c["target_abstain"]["delta"]
        out["heldout"][a]["abstain_rate"] = {k: float(np.mean(list(_by(r, k, "abstain").values()))) for k in C.ABS_KINDS + C.ANS_KINDS}
    out["heldout"]["qa_abstain_rate"] = {k: float(np.mean(list(_by(base, k, "abstain").values()))) for k in C.ABS_KINDS + C.ANS_KINDS}
    sel = reps["selected"]
    out["selected_minus_control"] = {}
    for a in reps:
        if a == "selected":
            continue
        o = reps[a]
        out["selected_minus_control"][a] = {
            "abstain_abs": metrics.paired_delta_ci(_by(sel, C.ABS_KINDS, "abstain"), _by(o, C.ABS_KINDS, "abstain")),
            "false_abstain_ans": metrics.paired_delta_ci(_by(sel, C.ANS_KINDS, "abstain"), _by(o, C.ANS_KINDS, "abstain")),
            "discrimination_min": discrimination(sel)["min"] - discrimination(o)["min"]}
    # full retention (both disjoint sets) vs θ_qa, paired over windows
    ms = [json.loads(l) for l in open(OUT / "log.jsonl", encoding="utf-8")]
    ms = [r for r in ms if r.get("kind") == "milestone"]
    out["milestone"] = {}
    for r in ms:
        row = {}
        for s in r["retention"]:
            k = s.replace("/", "__")
            a = np.load(OUT / "retention" / f"{r['arm']}__{k}.npy"); b = np.load(paths.RUNS / "gate1" / "retention" / f"qa__{k}.npy")
            row[s] = {"rel": float(a.mean() / b.mean() - 1), "delta": metrics.paired_delta_ci(dict(enumerate(a)), dict(enumerate(b)))}
        row["retention_pass"] = all(v["rel"] <= C.RETENTION_MAX_REL_INCREASE for v in row.values() if isinstance(v, dict))
        row["probes"] = {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))} for k, v in r["probes"].items()}
        out["milestone"][r["arm"]] = row
    (OUT / "gate3b_andgate_numbers.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()

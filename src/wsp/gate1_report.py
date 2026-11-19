"""Compute every Gate 1 number from saved artifacts -> wsp_runs/gate1/gate1_numbers.json + GATE1_TABLES.md.

Nothing here runs a model. Inputs: wsp_runs/gate1/{qa_all,e4_all_3shot}.json, retention/*.npy,
probes/*.json, splits/eval_overlap.json, data/reference_results/slm-125m-sft*.json.

    python -m wsp.gate1_report
"""
from __future__ import annotations

import json

import numpy as np

from . import checkpoints, metrics, paths

G = paths.RUNS / "gate1"
M = ("token_f1", "exact_match", "fabrication", "abstain", "false_abstain")
CONDS = ("clean", "realistic", "retrieval_failure", "closed_book")


def _load(p):
    return json.load(open(p, encoding="utf-8"))


def _by(items, cond=None):
    return {(i["cond"], i["pair_id"]): i for i in items if cond is None or i["cond"] == cond}


def _f(x, d=3):
    return f"{x:.{d}f}"


def _ci(t, d=3):
    return f"{_f(t[0], d)} [{_f(t[1], d)}, {_f(t[2], d)}]"


def _unpaired(a, b, n=5000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = rng.choice(a, (n, a.size)).mean(1) - rng.choice(b, (n, b.size)).mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": float(a.mean() - b.mean()), "ci": [float(lo), float(hi)],
            "significant": bool(lo > 0 or hi < 0), "n_a": int(a.size), "n_b": int(b.size)}


def main():
    out, md = {"lock": checkpoints.verify_lock()}, []
    qa = _load(G / "qa_all.json")
    e4 = _load(G / "e4_all_3shot.json")
    ref = _load(paths.REFERENCE / "slm-125m-sft.json")
    refj = {(i["cond"], i["pair_id"]): i for i in _load(paths.REFERENCE / "slm-125m-sft.judged.json")["per_item"]}
    qa_i, e4_i, ref_i = _by(qa["per_item"]), _by(e4["per_item"]), _by(ref["per_item"])

    # ---- 1. reproduction of the published clean numbers ----
    clean_keys = sorted(k for k in ref_i if k[0] == "clean")
    same = [k for k in clean_keys if qa_i[k]["resp"] == ref_i[k]["resp"]]
    rep = {"n": len(clean_keys), "byte_identical": len(same),
           "byte_identical_rate": len(same) / len(clean_keys), "metrics": {}}
    for m in ("token_f1", "exact_match", "fabrication", "false_abstain"):
        mine = metrics.bootstrap_ci(qa_i[k]["scores"][m] for k in clean_keys)
        pub = metrics.bootstrap_ci(ref_i[k]["scores"][m] for k in clean_keys)
        pd = metrics.paired_delta_ci({k: qa_i[k]["scores"][m] for k in clean_keys},
                                     {k: ref_i[k]["scores"][m] for k in clean_keys})
        rep["metrics"][m] = {"ours": mine, "published": pub, "paired_delta": pd,
                             "point_inside_published_ci": pub[1] <= mine[0] <= pub[2]}
    rep["PASS"] = all(v["point_inside_published_ci"] and not v["paired_delta"]["significant"]
                      for v in rep["metrics"].values())
    out["reproduction"] = rep
    md += ["## 1. Reproduction of θ_qa's published clean numbers", "",
           f"Byte-identical generations: **{len(same)}/{len(clean_keys)} ({rep['byte_identical_rate']:.1%})**", "",
           "| metric | ours | published | paired Δ (ours − published) | inside published CI |", "|---|---|---|---|---|"]
    for m, v in rep["metrics"].items():
        pd = v["paired_delta"]
        md.append(f"| {m} | {_ci(v['ours'])} | {_ci(v['published'])} | {pd['delta']:+.4f} [{pd['ci'][0]:+.4f}, {pd['ci'][1]:+.4f}] | {'yes' if v['point_inside_published_ci'] else 'NO'} |")
    md += ["", f"**Gate 1 reproduction: {'PASS' if rep['PASS'] else 'FAIL'}**", ""]

    # ---- 2. all four conditions, both originals ----
    out["conditions"] = {"qa": qa["result"]["conditions"], "e4_3shot": e4["result"]["conditions"]}
    md += ["## 2. All four conditions (deterministic metrics, n=500 each)", "",
           "| condition | model | token-F1 | exact match | fabrication ↓ | abstain rate |", "|---|---|---|---|---|---|"]
    for c in CONDS:
        for name, r in (("θ_e4 (3-shot)", e4), ("θ_qa", qa)):
            v = r["result"]["conditions"][c]
            md.append(f"| {c} | {name} | {_ci(v['token_f1'])} | {_ci(v['exact_match'])} | {_ci(v['fabrication'])} | {_ci(v['abstain_rate'])} |")
    md += ["", "Paired θ_qa − θ_e4 (same items):", "", "| condition | Δ token-F1 | Δ fabrication | Δ abstain |", "|---|---|---|---|"]
    out["qa_minus_e4"] = {}
    for c in CONDS:
        row = {}
        for m in ("token_f1", "fabrication", "abstain"):
            row[m] = metrics.paired_delta_ci({k: v["scores"][m] for k, v in qa_i.items() if k[0] == c},
                                             {k: v["scores"][m] for k, v in e4_i.items() if k[0] == c})
        out["qa_minus_e4"][c] = row
        md.append("| " + c + " | " + " | ".join(f"{row[m]['delta']:+.3f} [{row[m]['ci'][0]:+.3f}, {row[m]['ci'][1]:+.3f}]{' ✓' if row[m]['significant'] else ''}" for m in ("token_f1", "fabrication", "abstain")) + " |")
    md.append("")

    # ---- 3. train/eval overlap effect on θ_qa (clean) ----
    ov = _load(paths.SPLITS / "eval_overlap.json")["by_pair_id"]
    zero = [k for k in clean_keys if ov[k[1]] == 0]
    some = [k for k in clean_keys if ov[k[1]] > 0]
    oe = {}
    for m in ("token_f1", "exact_match", "fabrication"):
        oe[m] = _unpaired([qa_i[k]["scores"][m] for k in some], [qa_i[k]["scores"][m] for k in zero])
    jc = lambda k: (refj[k]["judge"]["correct"] - 1) / 4
    oe["judged_correctness_published"] = _unpaired([jc(k) for k in some], [jc(k) for k in zero])
    clean_only = metrics.bootstrap_ci(jc(k) for k in zero)
    oe["judged_correctness_zero_overlap_only"] = clean_only
    oe["judged_correctness_all"] = metrics.bootstrap_ci(jc(k) for k in clean_keys)
    out["overlap_effect"] = oe
    md += ["## 3. Does train/eval overlap inflate θ_qa's published scores?", "",
           f"Eval clean items with zero 13-gram overlap with qa_sft: **{len(zero)}**; with some overlap: **{len(some)}**.", "",
           "| metric | overlapping − clean items | significant |", "|---|---|---|"]
    for m in ("token_f1", "exact_match", "fabrication", "judged_correctness_published"):
        v = oe[m]
        md.append(f"| {m} | {v['delta']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] | {'yes' if v['significant'] else 'no'} |")
    md += ["", f"Judged correctness, all 500: {_ci(oe['judged_correctness_all'])}; zero-overlap 450 only: {_ci(clean_only)}", ""]

    # ---- 4. retention: forgetting from e4 -> qa ----
    from .retention import SETS
    ret = {}
    md += ["## 4. Retention: held-out loss, θ_e4 vs θ_qa (same windows, paired)", "",
           "| set / domain | windows | θ_e4 loss | θ_qa loss | Δ loss (qa − e4) | Δ % |", "|---|---|---|---|---|---|"]
    for s, srcs in SETS.items():
        for src in srcs:
            a = np.load(G / "retention" / f"e4__{s}__{src}.npy")
            b = np.load(G / "retention" / f"qa__{s}__{src}.npy")
            pd = metrics.paired_delta_ci(dict(enumerate(b)), dict(enumerate(a)))
            ret[f"{s}/{src}"] = {"e4": metrics.bootstrap_ci(a), "qa": metrics.bootstrap_ci(b), "delta": pd,
                                 "delta_pct": 100 * pd["delta"] / float(a.mean()),
                                 "per_window_delta_sd": float(np.std(b - a))}
            md.append(f"| {s}/{src} | {len(a)} | {_f(a.mean(), 4)} | {_f(b.mean(), 4)} | {pd['delta']:+.4f} [{pd['ci'][0]:+.4f}, {pd['ci'][1]:+.4f}] | {ret[f'{s}/{src}']['delta_pct']:+.2f}% |")
    out["retention"] = ret
    md.append("")

    # ---- 5. copy / induction probes ----
    pe, pq = _load(G / "probes" / "e4.json"), _load(G / "probes" / "qa.json")
    pr = {}
    md += ["## 5. Phase-0 copy / induction probes (paired: identical passages and codes)", "",
           "| probe | θ_e4 | θ_qa | paired Δ (qa − e4) |", "|---|---|---|---|"]
    for k in ("A_induction_random", "B_induction_natural"):
        md.append(f"| {k} (2nd-copy acc) | {_f(pe[k]['acc_second_copy'])} | {_f(pq[k]['acc_second_copy'])} | — |")
        pr[k] = {"e4": pe[k]["acc_second_copy"], "qa": pq[k]["acc_second_copy"]}
    for k in ("C_needle_verbatim", "C_control_no_needle", "C2_needle_question", "C2_control_no_needle"):
        a = {x["i"]: x["hit"] for x in pq[k]["per_item"]}
        b = {x["i"]: x["hit"] for x in pe[k]["per_item"]}
        pd = metrics.paired_delta_ci(a, b)
        pr[k] = {"e4": pe[k]["rate"], "qa": pq[k]["rate"], "delta": pd}
        md.append(f"| {k} | {_f(pe[k]['rate'])} {pe[k]['ci95']} | {_f(pq[k]['rate'])} {pq[k]['ci95']} | {pd['delta']:+.3f} [{pd['ci'][0]:+.3f}, {pd['ci'][1]:+.3f}]{' ✓' if pd['significant'] else ''} |")
    out["probes"] = pr
    md.append("")

    # ---- 6. noise floors (inputs for threshold calibration) ----
    nf = {m: (out["reproduction"]["metrics"][m]["ours"][2] - out["reproduction"]["metrics"][m]["ours"][1]) / 2
          for m in ("token_f1", "exact_match", "fabrication")}
    out["noise_floor_clean_ci_halfwidth"] = nf
    out["gpu_nondeterminism_flip_rate"] = 1 - rep["byte_identical_rate"]
    md += ["## 6. Noise floors", "",
           f"- 95% CI half-width at n=500 (clean, θ_qa): " + ", ".join(f"{m} ±{v:.3f}" for m, v in nf.items()),
           f"- Cross-hardware greedy flip rate (L4 → RTX 3060): {out['gpu_nondeterminism_flip_rate']:.1%} of items", ""]

    (G / "gate1_numbers.json").write_text(json.dumps(out, indent=1, default=float))
    (G / "GATE1_TABLES.md").write_text("# Gate 1 — computed tables\n\n_Generated by `python -m wsp.gate1_report` from saved artifacts._\n\n" + "\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

"""Gate 2 instrument self-checks — must all pass before any arm's result is trusted.

  1. zero-delta identity: materialize(θ_qa, 0) reproduces checkpoints.load("qa") exactly
  2. safety gate: passes the zero delta, vetoes a NaN delta and a 10x-magnitude delta
  3. fast retention subset (first N windows/domain) tracks the full held-out set
  4. θ_qa baseline on every iter_val fold (fast loop protocol), saved for paired comparisons

    python -m wsp.gate2_checks
"""
from __future__ import annotations

import json

import numpy as np

from . import batchgen, checkpoints, config_wsp as C, engine, loop_eval, paths
from .retention import per_window_loss

OUT = paths.RUNS / "gate2"


def main():
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoints.verify_lock()
    res = {}

    # ---- 3. fast retention subset vs full (no GPU: uses Gate 1's saved per-window losses) ----
    g1 = paths.RUNS / "gate1" / "retention"
    sub = {}
    for src in ("sec", "case-law", "fineweb-edu"):
        a, b = np.load(g1 / f"e4__e4_native__{src}.npy"), np.load(g1 / f"qa__e4_native__{src}.npy")
        n = C.RETENTION_FAST_WINDOWS
        full, fast = float(b.mean() / a.mean() - 1), float(b[:n].mean() / a[:n].mean() - 1)
        sub[src] = {"full_rel": full, "fast_rel": fast, "abs_error_pp": 100 * abs(fast - full)}
    res["fast_retention_vs_full"] = sub
    print("fast retention vs full (θ_e4→θ_qa forgetting):", json.dumps(sub, indent=1))

    # ---- 1. zero-delta identity ----
    ref_model, tok, eos = checkpoints.load("qa")
    base = engine.load_state("qa")
    zero = {k: torch.zeros_like(v) for k, v in base.items()}
    m0 = engine.materialize(base, zero)
    ids = loop_eval.retention_ids(64)
    l_ref = np.concatenate([per_window_loss(ref_model, x) for x in ids.values()])
    l_0 = np.concatenate([per_window_loss(m0, x) for x in ids.values()])
    g1qa = json.load(open(paths.RUNS / "gate1" / "qa_all.json", encoding="utf-8"))["per_item"]
    probe = [i for i in g1qa if i["cond"] == "clean"][:100]
    r_ref = batchgen.generate(ref_model, tok, eos, [(i["system"], i["user"]) for i in probe])
    r_0 = batchgen.generate(m0, tok, eos, [(i["system"], i["user"]) for i in probe])
    res["zero_delta_identity"] = {"max_abs_loss_diff": float(np.abs(l_ref - l_0).max()),
                                  "identical_generations": sum(a == b for a, b in zip(r_ref, r_0)), "n": len(probe)}
    print("zero-delta identity:", res["zero_delta_identity"])

    # ---- 2. safety gate ----
    canary = torch.from_numpy(np.concatenate([x[:C.CANARY_WINDOWS // 3 + 1] for x in loop_eval.retention_ids(64).values()]).astype(np.int64))
    gate = engine.SafetyGate(ref_model, canary)
    res["gate_zero"] = gate.check(zero, base, m0)
    del m0
    nan = {k: v.clone() for k, v in zero.items()}; k0 = next(iter(nan)); nan[k0][0, 0] = float("nan")
    res["gate_nan"] = gate.check(nan, base, ref_model)
    big = {k: 10 * v.clone() for k, v in base.items() if "mlp" in k}   # Δ = 10·W on every MLP matrix
    mb = engine.materialize(base, big)
    res["gate_big"] = gate.check(big, base, mb)
    del mb
    for k in ("gate_zero", "gate_nan", "gate_big"):
        print(k, {kk: res[k][kk] for kk in ("pass", "vetoes")})

    # ---- 4. θ_qa baseline on every iter_val fold ----
    ret = loop_eval.retention_ids()
    for f in range(4):
        rep = loop_eval.evaluate(ref_model, tok, f"iter_val_{f}", ret_ids=ret if f == 0 else None)
        (OUT / f"baseline_qa_iter_val_{f}.json").write_text(json.dumps(rep))
        a = rep["agg"]
        print(f"θ_qa iter_val_{f}: abstain abs1/2/3 " + " ".join(f"{a[k]['abstain'][0]:.3f}" for k in ("abs1", "abs2", "abs3"))
              + f"  F1 ans1/2/3 " + " ".join(f"{a[k]['token_f1'][0]:.3f}" for k in ("ans1", "ans2", "ans3"))
              + f"  ans1 fab {a['ans1']['fabrication'][0]:.3f}  n/kind={a['ans1']['n']}")
        res[f"baseline_iter_val_{f}"] = {k: {m: v[m] for m in v} for k, v in a.items()}

    ok = (res["zero_delta_identity"]["max_abs_loss_diff"] == 0.0 and res["gate_zero"]["pass"]
          and not res["gate_nan"]["pass"] and not res["gate_big"]["pass"])
    res["ALL_CHECKS_PASS"] = bool(ok)
    (OUT / "gate2_checks.json").write_text(json.dumps(res, indent=1, default=float))
    print("ALL_CHECKS_PASS:", ok)


if __name__ == "__main__":
    main()

"""Self-recursion study (docs/18, pre-registered). θ1 = docs/13 AND-gate edit; each recursive iteration
re-applies the AND-gate construction to the edited model, aimed at its own remaining failures on edit_dev,
at a fresh layer in {7, 8, 9}. GPU sections take the shared GPU lock.

    python -m wsp.self_recursion run
    python -m wsp.self_recursion report
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import and_gate_edit as AG
from . import batchgen, checkpoints, config_wsp as C, engine, metrics, neuron_edit as NE, paths, probes, retention, splits
from . import ovm_pool as O
from .editor_step import _gpu_lock
from .loop_eval import _by, compare
from .rank1_edit import _slice10, optimize_write

OUT = paths.RUNS / "gate6_recursion"
THETA1 = paths.RUNS / "gate3b_andgate" / "deltas" / "selected.safetensors"
CAND_LAYERS = (7, 8, 9)
RHOS = (0.5, 1.0)
N_SUB, N_GEN_ITEMS = 4000, 2000
MAX_K = 4
S_MAX, GATE_ROW_REL_MAX = 4.0, 0.2


def _gk(l, p):
    return f"model.layers.{l}.mlp.{p}.weight"


def _sub():
    items = splits.load("edit_dev")
    idx = sorted(np.random.default_rng(C.SEED).choice(len(items), N_SUB, replace=False).tolist())
    return [items[i] for i in idx]


def build(k, D, base, tok, eos):
    """Vectors for iteration k on θ_{k-1} = base + D. Returns {layer: info}; vectors saved to OUT/vectors/it{k}_L{l}.npz."""
    import torch
    vdir = OUT / "vectors"; vdir.mkdir(parents=True, exist_ok=True)
    mp = vdir / f"it{k}_meta.json"
    if mp.exists():
        return json.loads(mp.read_text())
    sub = _sub()
    m = engine.materialize(base, D)
    gpath = vdir / f"it{k}_subsample_gen.json"
    if gpath.exists():
        resp = json.loads(gpath.read_text())
    else:
        resp = batchgen.generate(m, tok, eos, [(it["messages"][0]["content"], it["messages"][1]["content"]) for it in sub], C.GEN_BATCH)
        gpath.write_text(json.dumps(resp))
    absent = np.array([it["kind"].startswith("abs") for it in sub], bool)
    ab = np.array([metrics.is_abstain(r) for r in resp]).astype(bool)
    fail = absent & ~ab
    layers = tuple(l for l in CAND_LAYERS if _gk(l, "gate_proj") not in D)
    meta = {"k": k, "n_sub": len(sub), "n_absent": int(absent.sum()), "n_fail": int(fail.sum()),
            "absent_abstain_rate_sub": float(ab[absent].mean()), "answerable_abstain_rate_sub": float(ab[~absent].mean()),
            "layers": {}}
    print(f"  it{k}: θ_{k-1} on edit_dev subsample: absent abstain {ab[absent].mean():.3f}, failures {fail.sum()}, "
          f"answerable abstain {ab[~absent].mean():.3f}; candidate layers {layers}", flush=True)
    if not layers or fail.sum() < 20:
        meta["stop"] = "no candidate layer" if not layers else "fewer than 20 failures"
        mp.write_text(json.dumps(meta, indent=1))
        return meta
    sl = np.array([_slice10(it) for it in sub])
    gidx = set(np.random.default_rng(C.SEED).choice(len(sub), N_GEN_ITEMS, replace=False).tolist())
    NE.LAYERS = layers; AG.LAYERS = layers
    X, XG, A = NE.collect(m, tok, sub, gidx)
    Xlast, St = AG.collect(m, tok, sub, gidx, sl)
    keep = fail | ~absent
    fail_items = [it for it, f in zip(sub, fail) if f]
    for l in layers:
        Xall = np.concatenate([XG[l], X[l]])
        c = np.linalg.solve(Xall.T @ Xall / len(Xall) + 1e-3 * np.eye(768), Xall.mean(0)).astype(np.float32)
        wd = (base[_gk(l, "down_proj")] + D.get(_gk(l, "down_proj"), 0)).numpy()
        imp = A[l] * np.linalg.norm(wd, axis=0)
        j = int(np.argmin(imp))
        w, pi = NE.fit_probe(X[l][keep], fail[keep].astype(float), XG[l], sl[keep], "real")
        wg = base[_gk(l, "gate_proj")] + D.get(_gk(l, "gate_proj"), 0)
        old, cap = wg[j].numpy(), GATE_ROW_REL_MAX * float(wg.norm())
        lo, hi = 0.0, S_MAX
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if np.linalg.norm(mid * w - old) <= cap else (lo, mid)
        S = lo
        wu = base[_gk(l, "up_proj")] + D.get(_gk(l, "up_proj"), 0)
        old_u, ucap = wu[j].double().cuda(), AG.UP_ROW_REL_MAX * float(wu.norm())
        cands = []
        for lr in AG.LAMBDA_REL:
            r_fit, _ = AG.ridge(St, l, "fit", lam_rel=lr)
            r_all, lam = AG.ridge(St, l, "all", lam_rel=lr)
            cands.append({"lambda_rel": lr, "slice_mse": AG.slice_mse(St, l, r_fit),
                          "up_row_rel": float((r_all - old_u).norm()) / float(wu.norm()), "admissible": float((r_all - old_u).norm()) <= ucap})
        ok = [cc for cc in cands if cc["admissible"]]
        info = {"neuron": j, "neuron_importance": float(imp[j]), "importance_median": float(np.median(imp)), "s": S,
                "probe": pi, "r_candidates": cands}
        if not ok:
            info["stop"] = "no admissible detector lambda"
            meta["layers"][str(l)] = info
            continue
        best = min(ok, key=lambda cc: cc["slice_mse"])
        r, _ = AG.ridge(St, l, "all", lam_rel=best["lambda_rel"])
        r = r.float().cpu().numpy()
        z = S * (Xlast[l] @ w)
        act = AG.silu(z) * (Xlast[l] @ r)
        fire = act[fail & (z > 0)]
        if len(fire) == 0 or not np.median(fire) > 0:
            info["stop"] = "detector does not fire on failures (m undefined)"
            meta["layers"][str(l)] = info
            continue
        info.update(r_lambda_rel=best["lambda_rel"], m=float(np.median(fire)) if len(fire) else float("nan"),
                    frac_fail_firing=float((fail & (act > AG.FIRE)).sum() / max(fail.sum(), 1)),
                    frac_answerable_firing=float((~absent & (act > AG.FIRE)).sum() / max((~absent).sum(), 1)),
                    frac_absent_ok_firing=float((absent & ab & (act > AG.FIRE)).sum() / max((absent & ab).sum(), 1)))
        vec = {"w": w, "r": r, "c": c}
        for rho in RHOS:
            d, wi = optimize_write(m, tok, fail_items, l, max_rel=rho)
            vec[f"delta_rho{rho:g}"] = d
            info[f"write_rho{rho:g}"] = wi
        np.savez(vdir / f"it{k}_L{l}.npz", **vec)
        meta["layers"][str(l)] = info
        print(f"  it{k} L{l}: neuron {j} S {S:.3f} probe {pi} m {info['m']:.3f} fail-firing {info['frac_fail_firing']:.3f} "
              f"answerable-firing {info['frac_answerable_firing']:.3f}", flush=True)
        torch.cuda.empty_cache()
    del m; torch.cuda.empty_cache()
    mp.write_text(json.dumps(meta, indent=1, default=float))
    return meta


def new_rows(k, l, rho, base, D, meta):
    import torch
    v = np.load(OUT / "vectors" / f"it{k}_L{l}.npz")
    info = meta["layers"][str(l)]
    j, S, m = info["neuron"], info["s"], info["m"]
    kg, ku, kd = _gk(l, "gate_proj"), _gk(l, "up_proj"), _gk(l, "down_proj")
    out = dict(D)
    dg, du, dd = torch.zeros_like(base[kg]), torch.zeros_like(base[ku]), torch.zeros_like(base[kd])
    dg[j] = S * torch.tensor(v["w"]) - base[kg][j]
    du[j] = torch.tensor(v["r"]) - base[ku][j]
    dd[:, j] = torch.tensor(v[f"delta_rho{rho:g}"]) / m - base[kd][:, j]
    out.update({kg: dg, ku: du, kd: dd})
    return out


def _heldout_ovm(name, delta, base, tok, eos):
    import torch
    gp = OUT / "ovm" / f"{name}.json"
    if gp.exists():
        return
    gp.parent.mkdir(parents=True, exist_ok=True)
    items = [it for it in O._load_items() if it["matched"]]
    with _gpu_lock():
        m = engine.materialize(base, delta)
        resp = batchgen.generate(m, tok, eos, [(i["messages"][0]["content"], i["messages"][1]["content"]) for i in items], C.GEN_BATCH)
        del m; torch.cuda.empty_cache()
    gp.write_text(json.dumps([{"id": it["id"], "kind": it["kind"], "seed": it["seed_raft_idx"], "resp": x,
                               "scores": metrics.score_item(x, it["messages"][2]["content"], it["messages"][1]["content"], not it["ovm_absent"])}
                              for it, x in zip(items, resp)]))


def _pooled(name):
    per = []
    for f in (1, 2, 3):
        p = (OUT / "reports" / f"{name}__iter_val_{f}.json") if name != "qa" else (paths.RUNS / "gate2" / f"baseline_qa_iter_val_{f}.json")
        per += json.loads(p.read_text())["per_item"]
    return {"per_item": per}


def _evaluate_model(R, name, D, base, tok, eos, meta):
    for f in (0, 1, 2, 3):
        with _gpu_lock():
            rec = R.run(name, D, f"iter_val_{f}", meta=meta)
    _heldout_ovm(name, D, base, tok, eos)
    return rec


def run():
    from .gate3_sweep import Runner
    OUT.mkdir(parents=True, exist_ok=True)
    with _gpu_lock():
        R = Runner(out=OUT, exp="gate6_recursion")
    base = R.base
    _, tok, eos = checkpoints.load("qa")
    chain = json.loads((OUT / "chain.json").read_text()) if (OUT / "chain.json").exists() else {"models": ["theta1"], "stop": None}
    D = engine.load_delta(THETA1)
    engine.save_delta(D, OUT / "deltas" / "theta1.safetensors")
    _evaluate_model(R, "theta1", D, base, tok, eos, {"theta": 1})
    for name in chain["models"][1:]:
        D = engine.load_delta(OUT / "deltas" / f"{name}.safetensors")
    k = len(chain["models"]) + 1
    while chain["stop"] is None and k <= MAX_K:
        prev = chain["models"][-1]
        prev_gain = R.res[f"{prev}__iter_val_0"]["compare"]["target_abstain"]["delta"]
        with _gpu_lock():
            meta = build(k, D, base, tok, eos)
        if "stop" in meta:
            chain["stop"] = f"it{k}: {meta['stop']}"; break
        recs = {}
        for l, info in meta["layers"].items():
            if "stop" in info:
                continue
            for rho in RHOS:
                nm = f"it{k}_L{l}_rho{rho:g}"
                with _gpu_lock():
                    recs[nm] = (R.run(nm, new_rows(k, int(l), rho, base, D, meta), "iter_val_0",
                                      meta={"k": k, "layer": int(l), "rho": rho}), int(l), rho)
        ok = {n: v for n, v in recs.items() if not v[0].get("vetoed") and v[0]["selectable"]
              and v[0]["compare"]["target_abstain"]["delta"] > prev_gain}
        R.log.write("selection", k=k, candidates={n: {"vetoed": bool(v[0].get("vetoed")), "selectable": v[0].get("selectable"),
                    "abstain_gain": v[0].get("compare", {}).get("target_abstain", {}).get("delta")} for n, v in recs.items()},
                    prev_gain=prev_gain, chosen=max(ok, key=lambda n: ok[n][0]["compare"]["target_abstain"]["delta"]) if ok else None)
        if not ok:
            chain["stop"] = f"it{k}: no improving edit (prev dev gain {prev_gain:+.3f})"; break
        ch = max(ok, key=lambda n: ok[n][0]["compare"]["target_abstain"]["delta"])
        _, l, rho = ok[ch]
        D = new_rows(k, l, rho, base, D, meta)
        name = f"theta{k}"
        engine.save_delta(D, OUT / "deltas" / f"{name}.safetensors")
        print(f"== it{k}: selected {ch} -> {name}", flush=True)
        _evaluate_model(R, name, D, base, tok, eos, {"theta": k, "from": ch})
        chain["models"].append(name); chain.setdefault("selected", {})[name] = ch
        (OUT / "chain.json").write_text(json.dumps(chain, indent=1))
        c = compare(_pooled(name), _pooled("qa"))
        if not c["guardrails_pass"]:
            chain["stop"] = f"it{k}: {name} fails a held-out guardrail (pooled) -> degradation"; break
        if k >= 3:
            sc = [compare({"per_item": json.loads((OUT / "reports" / f"{chain['models'][i]}__iter_val_{i - 1 + 1}.json").read_text())["per_item"]},
                          {"per_item": json.loads((OUT / "reports" / f"{chain['models'][i - 1]}__iter_val_{i - 1 + 1}.json").read_text())["per_item"]})
                  ["target_abstain"]["ci"] for i in (len(chain["models"]) - 2, len(chain["models"]) - 1)]
            if all(lo <= 0 <= hi for lo, hi in sc):
                chain["stop"] = f"it{k}: plateau (scoring-fold abstention gain CI includes 0 twice)"; break
        k += 1
    if chain["stop"] is None:
        chain["stop"] = f"reached MAX_K={MAX_K}"
    (OUT / "chain.json").write_text(json.dumps(chain, indent=1))
    print("stop:", chain["stop"], flush=True)
    import torch
    for name in chain["models"]:
        pp = OUT / "probes" / f"{name}.json"
        if pp.exists():
            continue
        Dn = engine.load_delta(OUT / "deltas" / f"{name}.safetensors")
        with _gpu_lock():
            m = engine.materialize(base, Dn)
            rr = retention.evaluate(m, OUT / "retention", name)
            pr = probes.run(m, R.tok)
            del m; torch.cuda.empty_cache()
        pp.parent.mkdir(parents=True, exist_ok=True)
        pp.write_text(json.dumps({"retention": rr, "copy": {kk: vv for kk, vv in pr["C_needle_verbatim"].items() if kk != "per_item"}}, default=float))
    print("done", flush=True)


def report():
    chain = json.loads((OUT / "chain.json").read_text())
    base = _pooled("qa")
    items = [it for it in O._load_items() if it["matched"]]
    res = json.loads((OUT / "results.json").read_text())
    out = {"chain": chain, "models": {}}
    prev = None
    for i, name in enumerate(chain["models"]):
        rep = _pooled(name)
        c = compare(rep, base)
        row = {k: c[k] for k in ("target_abstain", "ans1.token_f1", "ans_all.false_abstain", "ans1.fabrication", "discrimination", "guardrails_pass")}
        row["dev"] = {k: res[f"{name}__iter_val_0"]["compare"][k] for k in ("target_abstain", "ans1.token_f1", "guardrails", "guardrails_pass")}
        row["dev_retention_pct"] = {k: 100 * v["rel"] for k, v in res[f"{name}__iter_val_0"]["compare"].get("retention", {}).items()}
        Dn = engine.load_delta(OUT / "deltas" / f"{name}.safetensors")
        row["norm_global_rel"] = res[f"{name}__iter_val_0"]["norm"]
        row["n_tensors"] = len(Dn)
        row["abstain_rate"] = {k: float(np.mean(list(_by(rep, k, "abstain").values()))) for k in C.ABS_KINDS + C.ANS_KINDS}
        pr = json.loads((OUT / "probes" / f"{name}.json").read_text())
        ret = {}
        for s in pr["retention"]:
            kk = s.replace("/", "__")
            a = np.load(OUT / "retention" / f"{name}__{kk}.npy"); b = np.load(paths.RUNS / "gate1" / "retention" / f"qa__{kk}.npy")
            ret[s] = float(a.mean() / b.mean() - 1)
        row["retention_rel"] = ret; row["copy"] = pr["copy"]
        row["all_guardrails_heldout"] = bool(c["guardrails_pass"] and all(v <= C.RETENTION_MAX_REL_INCREASE for v in ret.values())
                                             and pr["copy"]["rate"] >= C.VERBATIM_COPY_MIN)
        per = json.loads((OUT / "ovm" / f"{name}.json").read_text())
        Dm = O._disc_ci(items, per)
        v2 = [p for p in rep["per_item"] if p["kind"] in O.ANS + O.ABS]
        Dv = O._disc_ci([{"ovm_absent": p["kind"].startswith("abs"), "seed_raft_idx": p["id"].split(":")[0], "kind": p["kind"]} for p in v2], v2)
        row["ovm"] = {"D_matched": Dm, "D_v2_k12": Dv, "D_del_only": O._disc_ci(items, per, O.DEL),
                      "abstain_rate": {k: float(np.mean([p["scores"]["abstain"] for p in per if p["kind"] == k])) for k in O.ANS + O.ABS + O.DEL},
                      "verdict": ("n/a" if Dv[0] < 0.05 else "BEYOND OVERLAP" if Dm[1] >= 0.5 * Dv[0]
                                  else "OVERLAP-DOMINATED" if Dm[2] < 0.25 * Dv[0] else "PARTIAL")}
        if prev is not None:
            prep = _pooled(prev)
            row["vs_prev_pooled"] = {k: compare(rep, prep)[k] for k in ("target_abstain", "ans_all.false_abstain", "ans1.token_f1")}
            f = i                                           # scoring fold of theta_{i+1} is iter_val_i
            row["scoring_fold"] = f"iter_val_{f}"
            row["vs_prev_scoring_fold"] = compare({"per_item": json.loads((OUT / "reports" / f"{name}__iter_val_{f}.json").read_text())["per_item"]},
                                                  {"per_item": json.loads((OUT / "reports" / f"{prev}__iter_val_{f}.json").read_text())["per_item"]})["target_abstain"]
            a = {p["id"]: p for p in prep["per_item"]}; b = {p["id"]: p for p in rep["per_item"]}
            right = lambda p: (p["scores"]["abstain"] if p["kind"].startswith("abs") else not p["scores"]["abstain"])
            won = [i_ for i_ in a if right(a[i_])]
            lost = [i_ for i_ in won if not right(b[i_])]
            newly = [i_ for i_ in a if not right(a[i_]) and right(b[i_])]
            row["regression"] = {"prev_right": len(won), "now_wrong": len(lost), "rate": len(lost) / max(len(won), 1),
                                 "newly_right": len(newly)}
        out["models"][name] = row
        prev = name
    t1, tK = out["models"]["theta1"], out["models"][chain["models"][-1]]
    K = len(chain["models"])
    if K < 2:
        out["outcome"] = "NO GAIN (no recursive iteration accepted)"
    elif not out["models"]["theta2"]["all_guardrails_heldout"]:
        out["outcome"] = "DEGRADES"
    else:
        good = [n for n in chain["models"] if out["models"][n]["all_guardrails_heldout"]]
        tK = out["models"][good[-1]]
        ab = compare(_pooled(good[-1]), _pooled("theta1"))["target_abstain"]
        out["last_good"] = good[-1]; out["last_good_vs_theta1_abstain"] = ab
        out["outcome"] = ("GENUINE CUMULATIVE GAIN" if tK["ovm"]["D_matched"][1] > t1["ovm"]["D_matched"][0]
                          else "ABSTENTION-ONLY GAIN" if ab["ci"][0] > 0 else "NO GAIN")
    (OUT / "gate6_recursion_numbers.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()

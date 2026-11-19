"""Gate 3b-iii — thresholded single-neuron abstention edit. Pre-registered in docs/12_GATE3B_NEURON_EDIT.md.

Pre-run amendment (before any dev evaluation; from an edit_dev smoke test): the gate sharpness s is
not fixed at 4 but set per layer as the largest s <= 4 with ||Δ gate row|| <= 0.2·||W_gate|| at τ = 1,
because s = 4 made the gate row 1.1x the whole matrix norm (certain safety veto).

Repurpose neuron j of block ℓ's MLP (three vectors replaced, nothing else touched):
    gate row  g_j = s · (w − τ·c)     w: no-intercept weighted logistic absence probe on x̂ (MLP input)
    up row    u_j = c                 c: near-constant direction, c·x̂ ≈ 1
    down col  d_j = δ / m             δ: norm-bounded write vector; m: median firing activation (absent)
Grid ℓ × ρ × τ on iter_val_0 → selection → held-out + random + structured controls + milestone.
All construction data from edit_dev only.

    python -m wsp.neuron_edit build
    python -m wsp.neuron_edit run
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import checkpoints, config_wsp as C, engine, paths, probes, retention, splits
from .act_probe import auroc
from .gate3_sweep import Runner
from .rank1_edit import _enc, _slice10, optimize_write

OUT = paths.RUNS / "gate3b_neuron"
LAYERS = (5, 6)
RHOS = (0.25, 0.5, 1.0)
TAUS = (0.0, 1.0)
S_MAX = 4.0
GATE_ROW_REL_MAX = 0.2      # pre-run amendment: ||Δ gate row|| <= 0.2·||W_gate|| (safety limit is 0.25)
LAMBDAS = (1e-4, 1e-3, 1e-2)
N_GENERIC_ITEMS, GENERIC_STRIDE = 2000, 8
SEED = C.SEED


def collect(model, tok, items, gidx, batch=16):
    """x̂ (gate_proj input) at the last token, x̂ at generic positions, and per-neuron mean |a| on generic."""
    import torch
    X = {l: np.zeros((len(items), 768), np.float32) for l in LAYERS}
    XG = {l: [] for l in LAYERS}
    A = {l: torch.zeros(3072, dtype=torch.float64, device="cuda") for l in LAYERS}
    na = 0
    cur = {}
    hooks = []
    for l in LAYERS:
        mlp = model.model.layers[l].mlp
        hooks.append(mlp.gate_proj.register_forward_pre_hook((lambda l: lambda m, i: cur.__setitem__(("x", l), i[0].detach()))(l)))
        hooks.append(mlp.down_proj.register_forward_pre_hook((lambda l: lambda m, i: cur.__setitem__(("a", l), i[0].detach()))(l)))
    enc = [_enc(tok, it) for it in items]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    with torch.no_grad():
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]; L = max(len(enc[i]) for i in idx)
            ids = torch.zeros((len(idx), L), dtype=torch.long); att = torch.zeros_like(ids)
            for r, i in enumerate(idx):
                ids[r, :len(enc[i])] = torch.tensor(enc[i]); att[r, :len(enc[i])] = 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(input_ids=ids.cuda(), attention_mask=att.cuda())
            for l in LAYERS:
                x, a = cur[("x", l)].float(), cur[("a", l)].float()
                for r, i in enumerate(idx):
                    n = len(enc[i])
                    X[l][i] = x[r, n - 1].cpu().numpy()
                    if i in gidx:
                        XG[l].append(x[r, :n - 1:GENERIC_STRIDE].cpu().numpy())
                        A[l] += a[r, :n - 1:GENERIC_STRIDE].abs().sum(0).double()
                        if l == LAYERS[0]:
                            na += len(range(0, n - 1, GENERIC_STRIDE))
    for h in hooks:
        h.remove()
    return X, {l: np.concatenate(XG[l]) for l in LAYERS}, {l: (A[l] / na).cpu().numpy() for l in LAYERS}


def logreg_noint(X, y, wt, lam, iters=300):
    """Weighted L2 logistic regression with NO intercept (the bias must come through c·x̂ ≈ 1)."""
    import torch
    Xt = torch.tensor(X, device="cuda"); yt = torch.tensor(y, device="cuda", dtype=torch.float32)
    wt = torch.tensor(wt / wt.sum(), device="cuda", dtype=torch.float32)
    w = torch.zeros(X.shape[1], device="cuda", requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        l = (wt * torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w, yt, reduction="none")).sum() + 0.5 * lam * (w @ w)
        l.backward()
        return l
    opt.step(closure)
    return w.detach().cpu().numpy()


def fit_probe(Xl, y_last, XG, sl, mode="real"):
    """mode: real | shuffled (labels permuted among last tokens) | all (every last token positive)."""
    if mode == "shuffled":
        y_last = np.random.default_rng(SEED).permutation(y_last)
    if mode == "all":
        y_last = np.ones_like(y_last)
    sd = np.concatenate([Xl, XG]).std(0) + 1e-6

    def design(mask):
        Xp = Xl[mask]; yp = y_last[mask]
        P = max(1.0, (yp == 1).sum()); Na = (yp == 0).sum(); Ng = len(XG)
        X = np.concatenate([Xp, XG]) / sd
        y = np.concatenate([yp, np.zeros(Ng)])
        wt = np.concatenate([np.where(yp == 1, 1.0, P / (2 * max(Na, 1))), np.full(Ng, P / (2 * Ng) if Na else P / Ng)])
        return X.astype(np.float32), y.astype(np.float32), wt.astype(np.float32)
    best = None
    for lam in LAMBDAS:
        w = logreg_noint(*design(~sl), lam)
        s = (Xl[sl] / sd) @ w
        yv = y_last[sl]
        a = auroc(s[yv == 1], s[yv == 0]) if (yv == 0).any() and (yv == 1).any() else float("nan")
        if best is None or (a == a and a > best[0]):
            best = (a, lam)
    w = logreg_noint(*design(np.ones(len(Xl), bool)), best[1]) / sd
    return w, {"lambda": best[1], "slice_auroc": best[0]}


def build():
    import torch
    checkpoints.verify_lock()
    (OUT / "vectors").mkdir(parents=True, exist_ok=True)
    model, tok, _ = checkpoints.load("qa")
    items = splits.load("edit_dev")
    y = np.array([1.0 if it["kind"].startswith("abs") else 0.0 for it in items])
    sl = np.array([_slice10(it) for it in items])
    gidx = set(np.random.default_rng(SEED).choice(len(items), N_GENERIC_ITEMS, replace=False).tolist())
    X, XG, A = collect(model, tok, items, gidx)
    base = engine.load_state("qa")
    absent = [it for it in items if it["kind"].startswith("abs")]
    meta = {"n_items": len(items), "n_generic_positions": {l: len(XG[l]) for l in LAYERS}, "s_max": S_MAX,
            "gate_row_rel_max": GATE_ROW_REL_MAX, "layers": {}}
    for l in LAYERS:
        Xall = np.concatenate([XG[l], X[l]])
        c = np.linalg.solve(Xall.T @ Xall / len(Xall) + 1e-3 * np.eye(768), Xall.mean(0)).astype(np.float32)
        cx = Xall @ c
        wd = base[f"model.layers.{l}.mlp.down_proj.weight"].numpy()
        imp = A[l] * np.linalg.norm(wd, axis=0)
        j = int(np.argmin(imp))
        info = {"c_mean": float(cx.mean()), "c_std": float(cx.std()), "neuron": j, "neuron_importance": float(imp[j]),
                "importance_median": float(np.median(imp))}
        vec = {"c": c}
        wg = base[f"model.layers.{l}.mlp.gate_proj.weight"]
        S = None
        for mode in ("real", "shuffled", "all"):
            w, pi = fit_probe(X[l], y, XG[l], sl, mode)
            if S is None:                                    # sharpness from the REAL probe, reused by controls
                old, cap = wg[j].numpy(), GATE_ROW_REL_MAX * float(wg.norm())
                lo, hi = 0.0, S_MAX
                for _ in range(40):
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if np.linalg.norm(mid * (w - 1.0 * c) - old) <= cap else (lo, mid)
                S = lo
                info["s"] = S
                info["gate_row_rel"] = float(np.linalg.norm(S * (w - c) - old) / float(wg.norm()))
            sc_last, sc_gen = X[l] @ w, XG[l] @ w
            info[f"probe_{mode}"] = {**pi, "generic_mean_abs_silu": float(np.abs((lambda z: z / (1 + np.exp(-z)))(S * (sc_gen - XG[l] @ c * 0))).mean()), "score_absent_mean": float(sc_last[y == 1].mean()),
                                    "score_answerable_mean": float(sc_last[y == 0].mean()),
                                    "score_generic_p99": float(np.percentile(sc_gen, 99)),
                                    "generic_frac_above_0": float((sc_gen > 0).mean())}
            vec[f"w_{mode}"] = w
            for tau in TAUS:                                 # median firing activation on absent items
                z = S * (sc_last - tau * (X[l] @ c))
                act = z / (1 + np.exp(-z)) * (X[l] @ c)
                fire = act[(y == 1) & (z > 0)]
                info[f"probe_{mode}"][f"tau{tau:g}"] = {"m": float(np.median(fire)) if len(fire) else float("nan"),
                                                       "frac_absent_firing": float(((y == 1) & (z > 0)).sum() / (y == 1).sum()),
                                                       "frac_answerable_firing": float(((y == 0) & (z > 0)).sum() / (y == 0).sum()),
                                                       "frac_generic_firing": float((S * (sc_gen - tau * (XG[l] @ c)) > 0).mean())}
        print(f"  L{l}: {json.dumps(info, default=float)}", flush=True)
        for rho in RHOS:
            d, wi = optimize_write(model, tok, absent, l, max_rel=rho)
            vec[f"delta_rho{rho:g}"] = d
            info[f"write_rho{rho:g}"] = wi
        meta["layers"][l] = info
        np.savez(OUT / "vectors" / f"L{l}.npz", **vec)
        (OUT / "vectors" / "build_meta.json").write_text(json.dumps(meta, indent=1, default=float))
        torch.cuda.empty_cache()
    print("build done", flush=True)


def edit(base, l, rho, tau, read="real", ablate_only=False):
    import torch
    v = np.load(OUT / "vectors" / f"L{l}.npz")
    meta = json.loads((OUT / "vectors" / "build_meta.json").read_text())["layers"][str(l)]
    j, S = meta["neuron"], meta["s"]
    kg, ku, kd = (f"model.layers.{l}.mlp.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj"))
    dg, du, dd = torch.zeros_like(base[kg]), torch.zeros_like(base[ku]), torch.zeros_like(base[kd])
    if ablate_only:
        dd[:, j] = -base[kd][:, j]
        return {kd: dd}
    c = torch.tensor(v["c"]); w = torch.tensor(v[f"w_{read}"])
    m = meta[f"probe_{read}"][f"tau{tau:g}"]["m"]
    dg[j] = S * (w - tau * c) - base[kg][j]
    du[j] = c - base[ku][j]
    dd[:, j] = torch.tensor(v[f"delta_rho{rho:g}"]) / m - base[kd][:, j]
    return {kg: dg, ku: du, kd: dd}


def run():
    R = Runner(out=OUT, exp="gate3b_neuron")
    R.log.write("start", lock=R.lock, layers=LAYERS, rhos=RHOS, taus=TAUS,
                build_meta=json.loads((OUT / "vectors" / "build_meta.json").read_text()))
    for l in LAYERS:
        for rho in RHOS:
            for tau in TAUS:
                R.run(f"L{l}_rho{rho:g}_tau{tau:g}", edit(R.base, l, rho, tau), meta={"layer": l, "rho": rho, "tau": tau})
    dev = {k: v for k, v in R.res.items() if k.endswith("__iter_val_0") and v["arm"].startswith("L") and not v.get("vetoed")}
    ok = {k: v for k, v in dev.items() if v["selectable"]}
    if not ok:
        R.log.write("selection", chosen=None, reason="no config passed safety + guardrails + validity on iter_val_0")
        print("NO CONFIG PASSED — stop; report the grid.", flush=True)
        return
    ck = max(ok, key=lambda k: ok[k]["compare"]["target_abstain"]["delta"])
    ch = ok[ck]; l, rho, tau = ch["meta"]["layer"], ch["meta"]["rho"], ch["meta"]["tau"]
    R.log.write("selection", chosen=ch["arm"], meta=ch["meta"], passing=[v["arm"] for v in ok.values()])
    print(f"== selected {ch['arm']}", flush=True)
    sel = edit(R.base, l, rho, tau)
    engine.save_delta(sel, OUT / "deltas" / "selected.safetensors")
    arms = {"selected": sel, "ctrl_shuffled_read": edit(R.base, l, rho, tau, "shuffled"),
            "ctrl_write_only": edit(R.base, l, rho, tau, "all"), "ctrl_ablation": edit(R.base, l, rho, tau, ablate_only=True)}
    for s in range(3):
        arms[f"random_s{s}"] = engine.random_like(sel, seed=s)
    for name, d in arms.items():
        if name != "selected":
            R.run(name, d, "iter_val_0", meta={"control_of": ch["arm"]})
    for name, d in arms.items():
        for f in (1, 2, 3):
            R.run(name, d, f"iter_val_{f}", meta={"of": ch["arm"]})
    print("== milestone", flush=True)
    done = {r.get("arm") for r in map(json.loads, open(R.log.path, encoding="utf-8")) if r.get("kind") == "milestone"}
    for name in ("selected", "random_s0", "ctrl_ablation"):
        if name in done:
            continue
        m = engine.materialize(R.base, arms[name])
        ret = retention.evaluate(m, OUT / "retention", name)
        pr = probes.run(m, R.tok)
        (OUT / "probes").mkdir(parents=True, exist_ok=True)
        (OUT / "probes" / f"{name}.json").write_text(json.dumps(pr, default=float))
        R.log.write("milestone", arm=name, retention=ret,
                    probes={k: {kk: vv for kk, vv in v.items() if kk != "per_item"} for k, v in pr.items()})
        print(f"{name} verbatim copy {pr['C_needle_verbatim']['rate']}", flush=True)
        del m
    R.log.write("done")
    print("done", flush=True)


if __name__ == "__main__":
    {"build": build, "run": run}[sys.argv[1]]()

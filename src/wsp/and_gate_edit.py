"""Gate 3b-iv — AND-gate neuron edit. Pre-registered in docs/13_GATE3B_AND_GATE_EDIT.md.

Identical to the docs/12 single-neuron edit except the up row: u_j = r, a ridge reply-start detector
(r·x̂ ≈ 1 at the last prompt token, ≈ 0 on prompt text, template tokens, reply tokens and raw document
text). The neuron becomes SiLU(absence gate) × (reply start). Neuron j, S, c, w (all read modes) and δ are
reused from wsp_runs/gate3b_neuron/vectors/. All construction data from edit_dev only.

    python -m wsp.and_gate_edit build
    python -m wsp.and_gate_edit run
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np

from . import checkpoints, config_wsp as C, engine, paths, probes, retention, splits
from .gate3_sweep import Runner
from .neuron_edit import GENERIC_STRIDE, N_GENERIC_ITEMS, OUT as NEURON_OUT
from .rank1_edit import _enc, _slice10

OUT = paths.RUNS / "gate3b_andgate"
LAYERS = (5, 6)
RHOS = (0.25, 0.5, 1.0)
TAUS = (0.0, 1.0)
LAMBDA_REL = (1e-4, 1e-3, 1e-2, 1e-1)
UP_ROW_REL_MAX = 0.2
REPLY_TOKENS, RAW_STRIDE, RAW_MAX = 32, 4, 1000
SPECIAL = {1, 4, 5, 6}                      # <|eos|> <|user|> <|assistant|> <|system|>
GROUPS = ("reply_start", "prompt_text", "template", "reply_tokens", "raw_text")
FIRE = 0.5                                  # "fires" = activation > 0.5 (as in the docs/12 diagnosis)
SEED = C.SEED


def silu(z):
    return z / (1 + np.exp(-z))


def raw_docs(it) -> str:
    u = it["messages"][1]["content"].split("\n\nQuestion:")[0]
    return "\n\n".join(p.strip() for p in re.split(r"\[Document \d+\]\n", u) if p.strip())


class Stats:
    """Per (layer, group, slice-flag): Gram X'X, X'y, Σy², n — plus stored slice positions for descriptives."""

    def __init__(self):
        import torch
        self.t = torch
        self.G, self.b, self.yy, self.n, self.keep = {}, {}, {}, {}, {}

    def add(self, l, g, x, y, in_slice):
        t = self.t
        k = (l, g, bool(in_slice))
        x = x.double()
        if k not in self.G:
            self.G[k] = t.zeros(768, 768, dtype=t.float64, device="cuda")
            self.b[k] = t.zeros(768, dtype=t.float64, device="cuda"); self.yy[k] = 0.0; self.n[k] = 0
        self.G[k] += x.T @ x
        self.b[k] += x.T @ t.full((len(x),), float(y), dtype=t.float64, device="cuda")
        self.yy[k] += float(y) ** 2 * len(x); self.n[k] += len(x)
        if in_slice:
            self.keep.setdefault((l, g), []).append(x.float().cpu().numpy())


def collect(model, tok, items, gidx, sl):
    import torch
    S = Stats()
    Xlast = {l: np.zeros((len(items), 768), np.float32) for l in LAYERS}
    cur = {}
    hooks = [model.model.layers[l].mlp.gate_proj.register_forward_pre_hook(
        (lambda l: lambda m, i: cur.__setitem__(l, i[0].detach()))(l)) for l in LAYERS]

    def fwd(seqs):
        L = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), L), dtype=torch.long); att = torch.zeros_like(ids)
        for r, s in enumerate(seqs):
            ids[r, :len(s)] = torch.tensor(s); att[r, :len(s)] = 1
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=ids.cuda(), attention_mask=att.cuda())
        return {l: cur[l].float() for l in LAYERS}

    def batched(idx, make, handle, batch=16):
        seqs = {i: make(i) for i in idx}
        order = sorted(idx, key=lambda i: len(seqs[i]))
        for s in range(0, len(order), batch):
            b = order[s:s + batch]
            xs = fwd([seqs[i] for i in b])
            for r, i in enumerate(b):
                handle(i, seqs[i], {l: xs[l][r] for l in LAYERS})

    def h_prompt(i, ids, xs):                                       # pass A: templated prompts, all items
        n = len(ids)
        tpl = [p for p in range(n - 1) if ids[p] in SPECIAL or (p > 0 and ids[p - 1] in SPECIAL)]
        for l in LAYERS:
            x = xs[l][:n]
            Xlast[l][i] = x[n - 1].cpu().numpy()
            S.add(l, "reply_start", x[n - 1:n], 1.0, sl[i])
            if i in gidx:
                S.add(l, "prompt_text", x[:n - 1:GENERIC_STRIDE], 0.0, sl[i])
                S.add(l, "template", x[tpl], 0.0, sl[i])
    print("  pass A: templated prompts", flush=True)
    batched(list(range(len(items))), lambda i: _enc(tok, items[i]), h_prompt)

    G = sorted(gidx)
    rep = {i: tok(items[i]["messages"][2]["content"], add_special_tokens=False).input_ids[:REPLY_TOKENS] for i in G}

    def h_reply(i, ids, xs):                                        # pass B: prompt + teacher-forced reply
        k = len(rep[i])
        for l in LAYERS:
            S.add(l, "reply_tokens", xs[l][len(ids) - k:len(ids)], 0.0, sl[i])
    print("  pass B: reply tokens", flush=True)
    batched(G, lambda i: _enc(tok, items[i])[-(1000 - REPLY_TOKENS):] + rep[i], h_reply)

    def h_raw(i, ids, xs):                                          # pass C: raw document text, no template
        for l in LAYERS:
            S.add(l, "raw_text", xs[l][:len(ids):RAW_STRIDE], 0.0, sl[i])
    print("  pass C: raw document text", flush=True)
    batched(G, lambda i: tok(raw_docs(items[i]), add_special_tokens=False).input_ids[:RAW_MAX], h_raw)
    for h in hooks:
        h.remove()
    return Xlast, S


def ridge(S, l, which, lam_rel=None, lam=None):
    """Equal total weight per group. which: 'fit' (non-slice), 'all'."""
    t = S.t
    A = t.zeros(768, 768, dtype=t.float64, device="cuda"); b = t.zeros(768, dtype=t.float64, device="cuda")
    for g in GROUPS:
        ks = [(l, g, False)] + ([(l, g, True)] if which == "all" else [])
        n = sum(S.n[k] for k in ks)
        for k in ks:
            A += S.G[k] / n; b += S.b[k] / n
    if lam is None:
        lam = lam_rel * float(t.trace(A)) / 768
    r = t.linalg.solve(A + lam * t.eye(768, dtype=t.float64, device="cuda"), b)
    return r, lam


def slice_mse(S, l, r):
    return float(sum((r @ S.G[(l, g, True)] @ r - 2 * r @ S.b[(l, g, True)] + S.yy[(l, g, True)]) / S.n[(l, g, True)]
                     for g in GROUPS))


def build():
    import torch
    checkpoints.verify_lock()
    (OUT / "vectors").mkdir(parents=True, exist_ok=True)
    nmeta = json.loads((NEURON_OUT / "vectors" / "build_meta.json").read_text())
    model, tok, _ = checkpoints.load("qa")
    items = splits.load("edit_dev")
    y = np.array([1.0 if it["kind"].startswith("abs") else 0.0 for it in items])
    sl = np.array([_slice10(it) for it in items])
    gidx = set(np.random.default_rng(SEED).choice(len(items), N_GENERIC_ITEMS, replace=False).tolist())  # = docs/12 sample
    Xlast, S = collect(model, tok, items, gidx, sl)
    base = engine.load_state("qa")
    meta = {"n_items": len(items), "groups": GROUPS,
            "n_positions": {l: {g: S.n.get((l, g, False), 0) + S.n.get((l, g, True), 0) for g in GROUPS} for l in LAYERS},
            "up_row_rel_max": UP_ROW_REL_MAX, "layers": {}}
    for l in LAYERS:
        nm = nmeta["layers"][str(l)]
        v = np.load(NEURON_OUT / "vectors" / f"L{l}.npz")
        j, Sg = nm["neuron"], nm["s"]
        wu = base[f"model.layers.{l}.mlp.up_proj.weight"]
        old_u, cap = wu[j].double().cuda(), UP_ROW_REL_MAX * float(wu.norm())
        cands = []
        for lr in LAMBDA_REL:
            r_fit, _ = ridge(S, l, "fit", lam_rel=lr)
            r_all, lam = ridge(S, l, "all", lam_rel=lr)
            rel = float((r_all - old_u).norm()) / float(wu.norm())
            cands.append({"lambda_rel": lr, "lambda": lam, "slice_mse": slice_mse(S, l, r_fit), "up_row_rel": rel,
                          "admissible": float((r_all - old_u).norm()) <= cap})
        ok = [c for c in cands if c["admissible"]]
        info = {"neuron": j, "s": Sg, "lambda_candidates": cands}
        if not ok:
            info["stop"] = "no admissible lambda"
            meta["layers"][l] = info
            print(f"  L{l}: NO ADMISSIBLE LAMBDA {json.dumps(cands)}", flush=True)
            continue
        best = min(ok, key=lambda c: c["slice_mse"])
        r_fit, _ = ridge(S, l, "fit", lam_rel=best["lambda_rel"])
        r, _ = ridge(S, l, "all", lam_rel=best["lambda_rel"])
        r, r_fit = r.float().cpu().numpy(), r_fit.float().cpu().numpy()
        info.update(chosen_lambda_rel=best["lambda_rel"], up_row_rel=best["up_row_rel"])
        c, w = v["c"], v["w_real"]
        desc = {}
        for g in GROUPS:                                  # slice descriptives (r fit without the slice)
            Xs = np.concatenate(S.keep[(l, g)])
            d = Xs @ r_fit
            z0 = Sg * (Xs @ w)
            a_new, a_old = silu(z0) * d, silu(z0) * (Xs @ c)
            desc[g] = {"n": len(Xs), "detector_mean": float(d.mean()), "detector_rms": float(np.sqrt((d ** 2).mean())),
                       "detector_p99": float(np.percentile(d, 99)), "detector_p01": float(np.percentile(d, 1)),
                       "fire_rate_tau0_new": float((a_new > FIRE).mean()), "fire_rate_tau0_docs12": float((a_old > FIRE).mean())}
        info["slice_descriptives"] = desc
        dl = Xlast[l] @ r
        for mode in ("real", "shuffled", "all"):
            wm = v[f"w_{mode}"]
            info[f"probe_{mode}"] = {}
            for tau in TAUS:
                z = Sg * (Xlast[l] @ wm - tau * (Xlast[l] @ c))
                act = silu(z) * dl
                fire = act[(y == 1) & (z > 0)]
                info[f"probe_{mode}"][f"tau{tau:g}"] = {
                    "m": float(np.median(fire)) if len(fire) else float("nan"),
                    "m_docs12": nm[f"probe_{mode}"][f"tau{tau:g}"]["m"],
                    "frac_absent_firing": float(((y == 1) & (act > FIRE)).mean() / (y == 1).mean()),
                    "frac_answerable_firing": float(((y == 0) & (act > FIRE)).mean() / (y == 0).mean())}
        print(f"  L{l}: {json.dumps(info, default=float)}", flush=True)
        np.savez(OUT / "vectors" / f"L{l}.npz", r=r)
        meta["layers"][l] = info
        (OUT / "vectors" / "build_meta.json").write_text(json.dumps(meta, indent=1, default=float))
        torch.cuda.empty_cache()
    (OUT / "vectors" / "build_meta.json").write_text(json.dumps(meta, indent=1, default=float))
    print("build done", flush=True)


def edit(base, l, rho, tau, read="real", ablate_only=False):
    import torch
    v = np.load(NEURON_OUT / "vectors" / f"L{l}.npz")
    r = torch.tensor(np.load(OUT / "vectors" / f"L{l}.npz")["r"])
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
    du[j] = r - base[ku][j]
    dd[:, j] = torch.tensor(v[f"delta_rho{rho:g}"]) / m - base[kd][:, j]
    return {kg: dg, ku: du, kd: dd}


def run():
    R = Runner(out=OUT, exp="gate3b_andgate")
    bm = json.loads((OUT / "vectors" / "build_meta.json").read_text())
    R.log.write("start", lock=R.lock, layers=LAYERS, rhos=RHOS, taus=TAUS, build_meta=bm)
    for l in LAYERS:
        if "stop" in bm["layers"][str(l)]:
            print(f"L{l}: skipped ({bm['layers'][str(l)]['stop']})", flush=True)
            continue
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

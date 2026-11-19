"""Gate 3b-ii — rank-1 read→write abstention edit. Pre-registered in docs/11_GATE3B_RANK1_EDIT.md (Part A).

    W_down(ℓ) += γ · δ_ℓ r_ℓᵀ
      r_ℓ : closed-form ridge read vector on the 3072-d MLP hidden (input of W_down) at the last prompt
            token: rᵀh ≈ 1 absent, 0 answerable, ≈ 0 at generic prompt positions (penalty w·E[(rᵀh)²])
      δ_ℓ : 768-d vector that, added to block ℓ's output at the last prompt token, maximises
            log P("not stated in the context<|eos|>") on edit_dev absent items (θ_qa frozen)
    Grid ℓ ∈ LAYERS × γ ∈ GAMMAS on iter_val_0 → selection → held-out + random + structured controls
    + milestone. All construction data comes from edit_dev only.

    python -m wsp.rank1_edit build     # vectors only (edit_dev), saved to wsp_runs/gate3b_rank1/vectors/
    python -m wsp.rank1_edit run       # grid, selection, report (resumable)
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

from . import checkpoints, config_wsp as C, engine, paths, probes, retention, splits
from .act_probe import auroc
from .gate3_sweep import Runner
from .qa_eval import render

OUT = paths.RUNS / "gate3b_rank1"
LAYERS = (5, 6, 7, 8)
GAMMAS = (0.5, 1.0, 2.0)
LAMBDA_REL = (1e-4, 1e-3, 1e-2, 1e-1)
W_GENERIC = 1.0
N_GENERIC_ITEMS, GENERIC_STRIDE = 2000, 4
TARGET = "not stated in the context"
SEED = C.SEED


def _slice10(it) -> bool:
    """Fixed 10% seed-level slice of edit_dev used only to choose λ."""
    return int(hashlib.sha256(f"rank1-lambda:{it['seed_raft_idx']}".encode()).hexdigest()[:8], 16) % 10 == 0


def _enc(tok, it):
    return tok(render(it["messages"][0]["content"], it["messages"][1]["content"], []), add_special_tokens=False).input_ids[-1000:]


def collect(model, tok, items, generic_idx: set, batch=16):
    """Last-token down_proj inputs per layer (N x 3072, fp32) and generic second moments (3072 x 3072)."""
    import torch
    H = {l: np.zeros((len(items), 3072), np.float32) for l in LAYERS}
    G = {l: torch.zeros((3072, 3072), dtype=torch.float64, device="cuda") for l in LAYERS}
    gn = 0
    cur = {}
    hooks = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(
        (lambda l: lambda mod, inp: cur.__setitem__(l, inp[0].detach()))(l)) for l in LAYERS]
    enc = [_enc(tok, it) for it in items]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    with torch.no_grad():
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]; L = max(len(enc[i]) for i in idx)
            ids = torch.zeros((len(idx), L), dtype=torch.long); att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                ids[r, :len(enc[i])] = torch.tensor(enc[i]); att[r, :len(enc[i])] = 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(input_ids=ids.cuda(), attention_mask=att.cuda())
            for l in LAYERS:
                h = cur[l].float()
                for r, i in enumerate(idx):
                    n = len(enc[i])
                    H[l][i] = h[r, n - 1].cpu().numpy()
                    if i in generic_idx:
                        g = h[r, :n - 1:GENERIC_STRIDE]
                        G[l] += (g.T @ g).double()
                        if l == LAYERS[0]:
                            gn += g.shape[0]
    for h in hooks:
        h.remove()
    return H, {l: (G[l] / gn).float().cpu().numpy() for l in LAYERS}, gn


def ridge(H, y, Cg, lam_rel):
    import torch
    Ht = torch.tensor(H, device="cuda", dtype=torch.float64)
    A = Ht.T @ Ht / len(H)
    lam = lam_rel * float(torch.trace(A)) / A.shape[0]
    A = A + W_GENERIC * torch.tensor(Cg, device="cuda", dtype=torch.float64) + lam * torch.eye(A.shape[0], device="cuda", dtype=torch.float64)
    b = Ht.T @ torch.tensor(y, device="cuda", dtype=torch.float64) / len(H)
    return torch.linalg.solve(A, b).float().cpu().numpy()


def fit_read(H, y, Cg, sl):
    best = None
    for lr in LAMBDA_REL:
        r = ridge(H[~sl], y[~sl], Cg, lr)
        s = H[sl] @ r
        a = auroc(s[y[sl] == 1], s[y[sl] == 0])
        if best is None or a > best[0]:
            best = (a, lr)
    return ridge(H, y, Cg, best[1]), {"lambda_rel": best[1], "slice_auroc": best[0]}


def optimize_write(model, tok, items, layer, steps=300, micro=8, accum=2, max_rel=None):
    """δ added to block `layer` output at the last prompt token; maximise teacher-forced log P(target)."""
    import torch
    rng = np.random.default_rng(SEED)
    tgt = tok(TARGET, add_special_tokens=False).input_ids + [1]
    enc = [_enc(tok, it) for it in items]
    # residual scale at that layer for the learning rate
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        hs = [model(input_ids=torch.tensor([e]).cuda(), output_hidden_states=True).hidden_states[layer + 1][0, -1].float()
              for e in enc[:32]]
    rms = float(torch.stack(hs).pow(2).mean().sqrt())
    delta = torch.zeros(model.config.hidden_size, device="cuda", requires_grad=True)
    opt = torch.optim.Adam([delta], lr=0.02 * rms)
    pos = {}

    def hook(mod, inp, out):
        h = out[0]
        add = torch.zeros_like(h)
        add[torch.arange(h.shape[0]), pos["p"]] = delta.to(h.dtype)
        return (h + add,) + tuple(out[1:])
    hd = model.model.layers[layer].register_forward_hook(hook)
    for p in model.parameters():
        p.requires_grad_(False)
    hist = []
    for step in range(steps):
        opt.zero_grad(); tot = 0.0
        for _ in range(accum):
            b = [enc[i] for i in rng.integers(0, len(enc), micro)]
            L = max(len(e) + len(tgt) for e in b)
            ids = torch.zeros((len(b), L), dtype=torch.long); att = torch.zeros_like(ids)
            lab = torch.full_like(ids, -100)
            for j, e in enumerate(b):
                ids[j, :len(e) + len(tgt)] = torch.tensor(e + tgt); att[j, :len(e) + len(tgt)] = 1
                lab[j, len(e):len(e) + len(tgt)] = torch.tensor(tgt)
            pos["p"] = torch.tensor([len(e) - 1 for e in b], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits.float()
            lp = torch.log_softmax(lg[:, :-1], -1)
            labs = lab[:, 1:].cuda(); m = labs != -100
            ll = lp.gather(-1, labs.clamp_min(0)[..., None])[..., 0]
            loss = -(ll * m).sum() / len(b) / accum
            loss.backward(); tot += loss.item()
        opt.step(); hist.append(-tot)
        if max_rel is not None:                         # projected Adam: ||δ|| <= max_rel * ||residual||
            with torch.no_grad():
                cap = max_rel * rms * model.config.hidden_size ** .5
                n = delta.norm()
                if n > cap:
                    delta.mul_(cap / n)
        if step % 50 == 0 or step == steps - 1:
            print(f"    δ L{layer} step {step} mean logP(target) {-tot:.3f} |δ|/|h| {float(delta.norm()) / (rms * model.config.hidden_size ** .5):.2f}", flush=True)
    hd.remove()
    d = delta.detach().cpu().numpy()
    return d, {"rms_residual": rms, "delta_norm": float(np.linalg.norm(d)),
               "delta_rel_to_residual": float(np.linalg.norm(d) / (rms * model.config.hidden_size ** .5)),
               "logp_start": hist[0], "logp_end": hist[-1]}


def build():
    import torch
    checkpoints.verify_lock()
    (OUT / "vectors").mkdir(parents=True, exist_ok=True)
    model, tok, _ = checkpoints.load("qa")
    items = splits.load("edit_dev")
    y = np.array([1.0 if it["kind"].startswith("abs") else 0.0 for it in items])
    sl = np.array([_slice10(it) for it in items])
    gidx = set(np.random.default_rng(SEED).choice(len(items), N_GENERIC_ITEMS, replace=False).tolist())
    print(f"collecting keys: {len(items)} items, generic from {len(gidx)} items", flush=True)
    H, G, gn = collect(model, tok, items, gidx)
    absent = [it for it in items if it["kind"].startswith("abs")]
    meta = {"n_items": len(items), "n_generic_positions": gn, "w_generic": W_GENERIC, "layers": {}}
    for l in LAYERS:
        r, info = fit_read(H[l], y, G[l], sl)
        r_shuf, _ = fit_read(H[l], np.random.default_rng(SEED).permutation(y), G[l], sl)
        r_all, _ = fit_read(H[l], np.ones_like(y), G[l], sl)
        s = H[l] @ r
        info.update({"mean_read_absent": float(s[y == 1].mean()), "mean_read_answerable": float(s[y == 0].mean()),
                     "rms_read_generic": float(np.sqrt(r @ G[l] @ r)),
                     "mean_read_shuffled_absent": float((H[l] @ r_shuf)[y == 1].mean()),
                     "mean_read_writeonly_absent": float((H[l] @ r_all)[y == 1].mean()),
                     "mean_read_writeonly_answerable": float((H[l] @ r_all)[y == 0].mean())})
        print(f"  L{l} read: {info}", flush=True)
        d, winfo = optimize_write(model, tok, absent, l)
        info["write"] = winfo
        meta["layers"][l] = info
        np.savez(OUT / "vectors" / f"L{l}.npz", r=r, r_shuf=r_shuf, r_all=r_all, delta=d)
        (OUT / "vectors" / "build_meta.json").write_text(json.dumps(meta, indent=1, default=float))
        torch.cuda.empty_cache()
    print("build done", flush=True)


def edit(l, gamma, read="r"):
    import torch
    v = np.load(OUT / "vectors" / f"L{l}.npz")
    return {f"model.layers.{l}.mlp.down_proj.weight": torch.tensor(gamma * np.outer(v["delta"], v[read]), dtype=torch.float32)}


def run():
    R = Runner(out=OUT, exp="gate3b_rank1")
    R.log.write("start", lock=R.lock, layers=LAYERS, gammas=GAMMAS,
                build_meta=json.loads((OUT / "vectors" / "build_meta.json").read_text()))
    for l in LAYERS:
        for g in GAMMAS:
            R.run(f"L{l}_g{g:g}", edit(l, g), meta={"layer": l, "gamma": g})
    dev = {k: v for k, v in R.res.items() if k.endswith("__iter_val_0") and v["arm"].startswith("L") and not v.get("vetoed")}
    ok = {k: v for k, v in dev.items() if v["selectable"]}
    if not ok:
        R.log.write("selection", chosen=None, reason="no config passed safety + guardrails + validity on iter_val_0")
        print("NO CONFIG PASSED — stop; report the grid.", flush=True)
        return
    ck = max(ok, key=lambda k: ok[k]["compare"]["target_abstain"]["delta"])
    ch = ok[ck]; l, g = ch["meta"]["layer"], ch["meta"]["gamma"]
    R.log.write("selection", chosen=ch["arm"], meta=ch["meta"], passing=[v["arm"] for v in ok.values()])
    print(f"== selected {ch['arm']}", flush=True)
    sel = edit(l, g)
    engine.save_delta(sel, OUT / "deltas" / "selected.safetensors")
    arms = {"selected": sel, "ctrl_shuffled_read": edit(l, g, "r_shuf"), "ctrl_write_only": edit(l, g, "r_all")}
    for s in range(3):
        arms[f"random_s{s}"] = engine.random_like(sel, seed=s)
    for name, d in arms.items():
        if name != "selected":
            R.run(name, d, "iter_val_0", meta={"control_of": ch["arm"]})
    for name, d in arms.items():
        for f in (1, 2, 3):
            R.run(name, d, f"iter_val_{f}", meta={"of": ch["arm"]})
    print("== milestone", flush=True)
    for name in ("selected", "random_s0"):
        if any(r.get("kind") == "milestone" and r.get("arm") == name
               for r in map(json.loads, open(R.log.path, encoding="utf-8"))):
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

"""Gate 3b-i — linear probe for "answer absent" in hidden states. Pre-registered in
docs/10_GATE3B_ACTIVATION_PROBE.md (Part A). Edits nothing.

  models   : qa (target), e4 (pre-SFT parent), rand (random init, same config, seed 1337)
  features : last prompt token (primary) and prompt mean (secondary), layers 0..12
  data     : train edit_dev, choose layer/λ on iter_val_0, report iter_val_1..3 pooled
  probe    : L2 logistic regression on standardised features
  metrics  : AUROC per document count k (abs_k vs ans_k) + pooled, cluster bootstrap over seeds
  controls : shuffled labels, lexical-overlap baseline, rand, e4

    python -m wsp.act_probe
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter

import numpy as np

from . import checkpoints, paths, splits
from .qa_eval import render

OUT = paths.RUNS / "gate3b_probe"
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
KINDS = ("ans1", "ans2", "ans3", "abs1", "abs2", "abs3")
HELD = ("iter_val_1", "iter_val_2", "iter_val_3")
SEED = 1337


# ---------------- AUROC + cluster bootstrap ----------------
def auroc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    x = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]                      # average ranks, ties shared
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def by_seed(items, score):
    """-> {k: (pos[S], neg[S])} aligned by seed, plus pooled [S,3] arrays."""
    idx = {(it["seed_raft_idx"], it["kind"]): i for i, it in enumerate(items)}
    seeds = sorted({it["seed_raft_idx"] for it in items})
    P = np.array([[score[idx[(s, f"abs{k}")]] for k in (1, 2, 3)] for s in seeds])
    N = np.array([[score[idx[(s, f"ans{k}")]] for k in (1, 2, 3)] for s in seeds])
    return P, N


def evaluate_scores(items, score, n_boot=1000):
    P, N = by_seed(items, score)
    rng = np.random.default_rng(0)
    out = {}
    for j, k in enumerate((1, 2, 3)):
        pt = auroc(P[:, j], N[:, j])
        bs = [auroc(P[b, j], N[b, j]) for b in (rng.integers(0, len(P), len(P)) for _ in range(n_boot))]
        out[f"k{k}"] = [pt, *np.percentile(bs, [2.5, 97.5]).tolist()]
    pt = auroc(P.ravel(), N.ravel())
    bs = []
    for _ in range(n_boot):
        b = rng.integers(0, len(P), len(P))
        bs.append(auroc(P[b].ravel(), N[b].ravel()))
    out["pooled"] = [pt, *np.percentile(bs, [2.5, 97.5]).tolist()]
    return out


# ---------------- features ----------------
def load_model(tag):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    if tag == "rand":
        torch.manual_seed(SEED)
        cfg = AutoConfig.from_pretrained(checkpoints.model_dir("qa"))
        m = AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).cuda().eval()
        return m
    m, _, _ = checkpoints.load(tag)
    return m


def extract(model, tok, items, batch=16):
    """Right-padded (causal attention: padding never affects real positions). Returns
    last [13, N, 768] and mean [13, N, 768], float16."""
    import torch
    enc = [tok(render(i["messages"][0]["content"], i["messages"][1]["content"], []),
               add_special_tokens=False).input_ids[-1024:] for i in items]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    nl = model.config.num_hidden_layers + 1
    last = np.zeros((nl, len(enc), model.config.hidden_size), np.float16); mean = np.zeros_like(last)
    with torch.no_grad():
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]; L = max(len(enc[i]) for i in idx)
            ids = torch.zeros((len(idx), L), dtype=torch.long); att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                ids[r, :len(enc[i])] = torch.tensor(enc[i]); att[r, :len(enc[i])] = 1
            ids, att = ids.cuda(), att.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hs = model(input_ids=ids, attention_mask=att, output_hidden_states=True).hidden_states
            lens = att.sum(1)
            for l, h in enumerate(hs):
                h = h.float()
                last[l, idx] = h[torch.arange(len(idx)), lens - 1].cpu().numpy()
                mean[l, idx] = ((h * att[..., None]).sum(1) / lens[:, None]).cpu().numpy()
    return {"last": last, "mean": mean}


# ---------------- probe ----------------
def fit_logreg(X, y, lam, iters=200):
    import torch
    X = torch.tensor(X, dtype=torch.float32, device="cuda"); y = torch.tensor(y, dtype=torch.float32, device="cuda")
    w = torch.zeros(X.shape[1], device="cuda", requires_grad=True); b = torch.zeros(1, device="cuda", requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + 0.5 * lam * (w @ w)
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach().cpu().numpy(), float(b.detach().cpu())


def standardise(Xtr, *others):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return [(Xtr - mu) / sd] + [(X - mu) / sd for X in others]


def labels(items):
    return np.array([1.0 if it["kind"].startswith("abs") else 0.0 for it in items])


def probe_family(F, Y, items, tag, pool):
    """Per layer: pick λ on iter_val_0 (pooled AUROC); pick the layer on iter_val_0; report held-out."""
    tr, dv = F["edit_dev"][pool], F["iter_val_0"][pool]
    ho = np.concatenate([F[s][pool] for s in HELD], axis=1)
    per_layer = []
    for l in range(tr.shape[0]):
        Xtr, Xdv, Xho = standardise(tr[l].astype(np.float32), dv[l].astype(np.float32), ho[l].astype(np.float32))
        best = None
        for lam in LAMBDAS:
            w, b = fit_logreg(Xtr, Y["edit_dev"], lam)
            a = auroc((Xdv @ w + b)[Y["iter_val_0"] == 1], (Xdv @ w + b)[Y["iter_val_0"] == 0])
            if best is None or a > best[0]:
                best = (a, lam, w, b)
        a, lam, w, b = best
        per_layer.append({"layer": l, "lambda": lam, "dev_auroc": a, "_w": w, "_b": b, "_Xho": Xho, "_Xtr": Xtr})
        print(f"  {tag}/{pool} layer {l:2d} λ={lam:g} dev AUROC {a:.3f}", flush=True)
    sel = max(per_layer, key=lambda r: r["dev_auroc"])
    score = sel["_Xho"] @ sel["_w"] + sel["_b"]
    held = evaluate_scores(items["held"], score)
    res = {"model": tag, "pool": pool, "selected_layer": sel["layer"], "lambda": sel["lambda"],
           "dev_auroc_by_layer": [round(r["dev_auroc"], 4) for r in per_layer], "held_out": held}
    if tag == "qa":                                       # shuffled-label control at the selected config
        yp = np.random.default_rng(SEED).permutation(Y["edit_dev"])
        w, b = fit_logreg(sel["_Xtr"], yp, sel["lambda"])
        res["shuffled_label_control"] = evaluate_scores(items["held"], sel["_Xho"] @ w + b, n_boot=200)
    print(f"  => {tag}/{pool} layer {sel['layer']}: held-out AUROC " +
          " ".join(f"{k} {v[0]:.3f}" for k, v in held.items()), flush=True)
    return res


# ---------------- lexical baseline ----------------
def lexical(items_by_split):
    df = Counter()
    docs = {d for it in items_by_split["edit_dev"] for d in splits.docs_of(it["messages"][1]["content"])}
    for d in docs:
        df.update(set(splits.words(d)))
    N = len(docs)
    idf = lambda w: math.log(N / (1 + df.get(w, 0)))

    def feats(it):
        u = it["messages"][1]["content"]; q = set(splits.words(splits.question_of(u))); ds = splits.docs_of(u)
        if not q:
            q = {"_"}
        tot = sum(idf(w) for w in q)
        rare = max(q, key=idf)
        cov = [sum(idf(w) for w in q if w in set(splits.words(d))) / tot for d in ds]
        frac = [len(q & set(splits.words(d))) / len(q) for d in ds]
        return [max(cov), float(np.mean(cov)), max(frac), float(any(re.search(rf"\b{re.escape(rare)}\b", d.lower()) for d in ds)),
                len(q), len(u)]
    return {s: np.array([feats(it) for it in its], np.float32) for s, its in items_by_split.items()}


def main():
    import torch
    checkpoints.verify_lock()
    OUT.mkdir(parents=True, exist_ok=True)
    sp = ("edit_dev", "iter_val_0") + HELD
    its = {s: splits.load(s, kinds=list(KINDS)) for s in sp}
    held_items = [i for s in HELD for i in its[s]]
    items = {"held": held_items}
    Y = {s: labels(its[s]) for s in sp}
    print("items:", {s: len(v) for s, v in its.items()}, flush=True)
    results = {"note": "pre-registered in docs/10_GATE3B_ACTIVATION_PROBE.md Part A", "n": {s: len(v) for s, v in its.items()}}

    # lexical baseline (no model)
    L = lexical(its)
    Xtr, Xdv, Xho = standardise(L["edit_dev"], L["iter_val_0"], np.concatenate([L[s] for s in HELD]))
    best = max(((auroc(*(lambda sc: (sc[Y["iter_val_0"] == 1], sc[Y["iter_val_0"] == 0]))(Xdv @ w + b)), lam, w, b)
                for lam in LAMBDAS for w, b in [fit_logreg(Xtr, Y["edit_dev"], lam)]), key=lambda t: t[0])
    results["lexical"] = {"lambda": best[1], "dev_auroc": best[0], "held_out": evaluate_scores(held_items, Xho @ best[2] + best[3])}
    print("lexical baseline held-out:", {k: round(v[0], 3) for k, v in results["lexical"]["held_out"].items()}, flush=True)

    _, tok, _ = checkpoints.load("qa")
    for tag in ("qa", "e4", "rand"):
        print(f"== {tag}", flush=True)
        m = load_model(tag)
        F = {s: extract(m, tok, its[s]) for s in sp}
        del m; torch.cuda.empty_cache()
        for pool in ("last", "mean"):
            results[f"{tag}/{pool}"] = probe_family(F, Y, items, tag, pool)
        del F
        (OUT / "probe_results.json").write_text(json.dumps(results, indent=1, default=float))

    q = results["qa/last"] if results["qa/last"]["dev_auroc_by_layer"][results["qa/last"]["selected_layer"]] >= \
        results["qa/mean"]["dev_auroc_by_layer"][results["qa/mean"]["selected_layer"]] else results["qa/mean"]
    ks = [q["held_out"][f"k{k}"][0] for k in (1, 2, 3)]
    verdict = "PRESENT" if min(ks) >= 0.80 else "WEAK" if min(ks) >= 0.65 else "ABSENT"
    results["decision"] = {"selected": f"qa/{q['pool']}", "layer": q["selected_layer"], "per_k_auroc": ks, "verdict": verdict}
    (OUT / "probe_results.json").write_text(json.dumps(results, indent=1, default=float))
    print("DECISION:", results["decision"], flush=True)


if __name__ == "__main__":
    main()

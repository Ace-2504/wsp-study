"""Overlap-matched (OVM) control pool. Pre-registered in docs/14_OVERLAP_MATCHED_POOL.md.

Question: does the installed abstention read answer-absence, or only word overlap between the question
and the documents? On pool v2 a 6-feature lexical-overlap model separates absent from answerable items
at AUROC 0.966 (docs/10), so pool v2 cannot tell the two apart.

Construction (per seed, from the seed's own pool-v2 items; no new documents):
  ans1 [g]            ans2 golden+D1 (pool v2 order)        -> answerable
  abs1 [D1]           abs2 D1+D2                            -> absent, topical distractors (pool v2)
  del1 [g_del]        del2 ans2 with g replaced by g_del    -> absent, ANSWER SENTENCE DELETED
g_del = the golden passage with every sentence overlapping the quoted support span removed. It keeps
the topic and most of the question's words but no longer contains the answer.
Matching: lexical-model logit (docs/10 lexical baseline, refit identically on pool-v2 edit_dev), 20
quantile bins per document count k ∈ {1,2}; in each bin draw min(#absent, #answerable) of each.

    python -m wsp.ovm_pool feasibility     # edit_dev seeds only: yields + score distributions
    python -m wsp.ovm_pool build           # held-out seeds (iter_val_1..3): pool + matching + audit sample
    python -m wsp.ovm_pool eval            # generate + score: θ_qa, AND-gate edit, Gate 2 LoRA lr3e-4
    python -m wsp.ovm_pool probe           # θ_qa layer-5 probe and the AND-gate read on the matched pool
    python -m wsp.ovm_pool report
"""
from __future__ import annotations

import collections
import json
import re
import sys

import numpy as np

from . import batchgen, checkpoints, config_wsp as C, engine, metrics, paths, splits
from .act_probe import auroc, fit_logreg, lexical, standardise

OUT = paths.RUNS / "ovm_pool"
HELD = ("iter_val_1", "iter_val_2", "iter_val_3")
N_BINS = 20
MIN_WORDS = 30
LEX_LAMBDA = 1e-4                      # docs/10 lexical baseline's selected λ
SEED = C.SEED
ANS, ABS, DEL = ("ans1", "ans2"), ("abs1", "abs2"), ("del1", "del2")
K_OF = {"ans1": 1, "abs1": 1, "del1": 1, "ans2": 2, "abs2": 2, "del2": 2}


def _raft():
    return [json.loads(l) for l in open(paths.RAFT, encoding="utf-8")]


def delete_answer(g: str, quote: str, answer: str):
    """-> (g_del, reason). reason is None when the item is usable."""
    s = g.find(quote)
    if s < 0:
        s = g.find(quote[:60])
    if s < 0:
        return None, "quote_not_found"
    e = s + len(quote)
    keep = []
    for m in re.finditer(r"[^.!?]*(?:[.!?]+|$)\s*", g):
        if not m.group().strip():
            continue
        if m.end() <= s or m.start() >= e:
            keep.append(m.group().strip())
    gd = " ".join(keep)
    if len(splits.words(gd)) < MIN_WORDS:
        return None, "too_short_after_deletion"
    na = splits.norm(answer)
    if (quote[:60] and quote[:60] in gd) or (len(na) >= 4 and na in splits.norm(gd)):
        return None, "answer_string_remains"
    aw = set(splits.words(answer))
    if aw and aw <= set(splits.words(gd)):
        return None, "all_answer_words_remain"
    nums = set(re.findall(r"\d[\d,.]*\d|\d", answer))
    if nums & set(re.findall(r"\d[\d,.]*\d|\d", gd)):
        return None, "answer_number_remains"
    return gd, None


def make_items(split_names):
    rf = _raft()
    by_seed = collections.defaultdict(dict)
    for s in split_names:
        for it in splits.load(s):
            by_seed[it["seed_raft_idx"]][it["kind"]] = it
    items, drops = [], collections.Counter()
    for sid, kinds in sorted(by_seed.items()):
        tgt = rf[sid]["messages"][2]["content"]
        quote = tgt.split("##begin_quote##")[1].split("##end_quote##")[0].strip()
        answer = tgt.rsplit("##end_quote##", 1)[1].strip()
        g = splits.docs_of(kinds["ans1"]["messages"][1]["content"])[0]
        gd, why = delete_answer(g, quote, answer)
        if why:
            drops[why] += 1
            continue
        q = splits.question_of(kinds["ans1"]["messages"][1]["content"])
        d2 = [gd if d == g else d for d in splits.docs_of(kinds["ans2"]["messages"][1]["content"])]
        assert len(d2) == 2 and gd in d2
        for k in ANS + ABS:
            items.append({**kinds[k], "ovm_absent": k.startswith("abs")})
        for kind, docs in (("del1", [gd]), ("del2", d2)):
            base = kinds["ans1"]
            items.append({**{x: base[x] for x in ("seed_raft_idx", "chunk_id", "source", "split")}, "kind": kind,
                          "id": f"{sid}:{kind}", "ovm_absent": True,
                          "messages": [base["messages"][0], {"role": "user", "content": splits.render_user(docs, q)},
                                       {"role": "assistant", "content": splits.ABSTAIN}]})
    return items, dict(drops), len(by_seed)


def lexical_model():
    """docs/10 lexical baseline, refit identically: features on pool-v2 edit_dev, standardised, λ = 1e-4."""
    ed = splits.load("edit_dev", kinds=list(C.ANS_KINDS + C.ABS_KINDS))
    y = np.array([1.0 if it["kind"].startswith("abs") else 0.0 for it in ed])
    F = lexical({"edit_dev": ed})["edit_dev"]
    mu, sd = F.mean(0), F.std(0) + 1e-6
    w, b = fit_logreg((F - mu) / sd, y, LEX_LAMBDA)

    def score(items):
        X = lexical({"edit_dev": ed, "x": items})["x"]
        return ((X - mu) / sd) @ w + b
    return score


def match(items, score):
    rng = np.random.default_rng(SEED)
    chosen = []
    for k in (1, 2):
        idx = [i for i, it in enumerate(items) if K_OF[it["kind"]] == k]
        s = score[idx]
        edges = np.quantile(s, np.linspace(0, 1, N_BINS + 1))
        b = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, N_BINS - 1)
        for j in range(N_BINS):
            pos = [idx[t] for t in np.where(b == j)[0] if items[idx[t]]["ovm_absent"]]
            neg = [idx[t] for t in np.where(b == j)[0] if not items[idx[t]]["ovm_absent"]]
            n = min(len(pos), len(neg))
            if n:
                chosen += rng.choice(pos, n, replace=False).tolist() + rng.choice(neg, n, replace=False).tolist()
    return sorted(chosen)


def cluster_auroc(items, sc, n_boot=1000):
    y = np.array([it["ovm_absent"] for it in items])
    seeds = np.array([it["seed_raft_idx"] for it in items])
    us = np.unique(seeds)
    grp = {s: np.where(seeds == s)[0] for s in us}
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(n_boot):
        ii = np.concatenate([grp[s] for s in rng.choice(us, len(us))])
        bs.append(auroc(sc[ii][y[ii]], sc[ii][~y[ii]]))
    return [auroc(sc[y], sc[~y]), *np.percentile(bs, [2.5, 97.5]).tolist()]


def describe(items, sc):
    out = {}
    for kind in ANS + ABS + DEL:
        v = np.array([sc[i] for i, it in enumerate(items) if it["kind"] == kind])
        if len(v):
            out[kind] = {"n": len(v), "mean": float(v.mean()), "p10": float(np.percentile(v, 10)),
                         "p50": float(np.median(v)), "p90": float(np.percentile(v, 90))}
    return out


def feasibility():
    items, drops, n = make_items(["edit_dev"])
    sc = lexical_model()(items)
    sel = match(items, sc)
    m = [items[i] for i in sel]
    res = {"seeds": n, "drops": drops, "seeds_kept": len(items) // 6, "lexical_logit": describe(items, sc),
           "matched": {"n": len(m), "by_kind": collections.Counter(it["kind"] for it in m),
                       "lexical_auroc": cluster_auroc(m, sc[sel])},
           "unmatched_lexical_auroc_ans_vs_del": cluster_auroc([it for it in items if it["kind"] in ANS + DEL],
                                                              sc[[i for i, it in enumerate(items) if it["kind"] in ANS + DEL]])}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "feasibility_edit_dev.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


def _load_items():
    return [json.loads(l) for l in open(OUT / "ovm_items.jsonl", encoding="utf-8")]


def build():
    checkpoints.verify_lock()
    items, drops, n = make_items(list(HELD))
    sc = lexical_model()(items)
    sel = set(match(items, sc))
    for i, it in enumerate(items):
        it["lex_logit"] = float(sc[i]); it["matched"] = i in sel
    m = [it for it in items if it["matched"]]
    msc = np.array([it["lex_logit"] for it in m])
    chk = cluster_auroc(m, msc)
    valid = (chk[1] <= 0.5 <= chk[2]) or (0.45 <= chk[0] <= 0.55)
    with open(OUT / "ovm_items.jsonl", "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")
    # blinded audit sample: 60 matched del1 + 30 matched ans1
    rng = np.random.default_rng(SEED)
    d1 = [it for it in m if it["kind"] == "del1"]; a1 = [it for it in m if it["kind"] == "ans1"]
    samp = [d1[i] for i in rng.choice(len(d1), 60, replace=False)] + [a1[i] for i in rng.choice(len(a1), 30, replace=False)]
    samp = [samp[i] for i in rng.permutation(len(samp))]
    key = {}
    with open(OUT / "audit_sample.jsonl", "w", encoding="utf-8") as fh:
        for j, it in enumerate(samp):
            u = it["messages"][1]["content"]
            fh.write(json.dumps({"aid": f"A{j:03d}", "question": splits.question_of(u), "passage": splits.docs_of(u)[0]}) + "\n")
            key[f"A{j:03d}"] = {"id": it["id"], "kind": it["kind"]}
    (OUT / "audit_key.json").write_text(json.dumps(key, indent=1))
    res = {"seeds": n, "drops": drops, "seeds_kept": len(items) // 6, "n_items": len(items),
           "lexical_logit": describe(items, sc),
           "matched": {"n": len(m), "by_kind": collections.Counter(it["kind"] for it in m), "lexical_auroc": chk,
                       "pool_valid": valid},
           "unmatched_lexical_auroc_v2_k12": cluster_auroc([it for it in items if it["kind"] in ANS + ABS],
                                                          sc[[i for i, it in enumerate(items) if it["kind"] in ANS + ABS]])}
    (OUT / "build.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


MODELS = {"qa": None, "andgate": paths.RUNS / "gate3b_andgate" / "deltas" / "selected.safetensors",
          "lora_lr3e-4": paths.RUNS / "gate2" / "deltas" / "lora_lr0.0003.safetensors",
          "fullft": paths.RUNS / "gate4_fullft" / "deltas" / "selected.safetensors"}


def eval_models():
    import torch
    checkpoints.verify_lock()
    items = [it for it in _load_items() if it["matched"]]
    _, tok, eos = checkpoints.load("qa")
    base = engine.load_state("qa")
    for name, dp in MODELS.items():
        f = OUT / f"gen_{name}.json"
        if f.exists():
            continue
        model = engine.materialize(base, engine.load_delta(dp) if dp else {})
        resp = batchgen.generate(model, tok, eos, [(i["messages"][0]["content"], i["messages"][1]["content"]) for i in items], C.GEN_BATCH)
        per = [{"id": it["id"], "kind": it["kind"], "seed": it["seed_raft_idx"], "resp": r,
                "scores": metrics.score_item(r, it["messages"][2]["content"], it["messages"][1]["content"], not it["ovm_absent"])}
               for it, r in zip(items, resp)]
        f.write_text(json.dumps(per))
        print(f"{name}: abstain absent {np.mean([p['scores']['abstain'] for p, it in zip(per, items) if it['ovm_absent']]):.3f} "
              f"answerable {np.mean([p['scores']['abstain'] for p, it in zip(per, items) if not it['ovm_absent']]):.3f}", flush=True)
        del model; torch.cuda.empty_cache()


def probe():
    import torch
    from .act_probe import extract
    from .neuron_edit import OUT as NEURON_OUT
    from .and_gate_edit import OUT as AG_OUT
    checkpoints.verify_lock()
    m_items = [it for it in _load_items() if it["matched"]]
    model, tok, _ = checkpoints.load("qa")
    ed = splits.load("edit_dev", kinds=list(C.ANS_KINDS + C.ABS_KINDS))
    y = np.array([1.0 if it["kind"].startswith("abs") else 0.0 for it in ed])
    Ftr = extract(model, tok, ed)["last"][5].astype(np.float32)
    Fm = extract(model, tok, m_items)["last"][5].astype(np.float32)
    Xtr, Xm = standardise(Ftr, Fm)
    w, b = fit_logreg(Xtr, y, 1e-2)
    res = {"qa_layer5_probe_matched_auroc": cluster_auroc(m_items, Xm @ w + b)}
    # the AND-gate edit's own read at block 6 (x̂ = gate_proj input; unaffected by the block-6 MLP edit)
    from .rank1_edit import _enc
    v = np.load(NEURON_OUT / "vectors" / "L6.npz"); r = np.load(AG_OUT / "vectors" / "L6.npz")["r"]
    S = json.loads((NEURON_OUT / "vectors" / "build_meta.json").read_text())["layers"]["6"]["s"]
    X = np.zeros((len(m_items), 768), np.float32); cur = {}
    h = model.model.layers[6].mlp.gate_proj.register_forward_pre_hook(lambda mod, i: cur.__setitem__("x", i[0].detach()))
    with torch.no_grad():
        for i, it in enumerate(m_items):
            ids = torch.tensor([_enc(tok, it)]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(input_ids=ids)
            X[i] = cur["x"][0, -1].float().cpu().numpy()
    h.remove()
    z = S * (X @ v["w_real"])
    act = z / (1 + np.exp(-z)) * (X @ r)
    res["andgate_read_matched_auroc"] = cluster_auroc(m_items, z)
    res["andgate_neuron_activation_matched_auroc"] = cluster_auroc(m_items, act)
    for kind in ANS + ABS + DEL:
        sel = np.array([it["kind"] == kind for it in m_items])
        res.setdefault("andgate_fire_rate_by_kind", {})[kind] = float((act[sel] > 0.5).mean()) if sel.any() else None
    (OUT / "probe.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


def _disc_ci(items, per, absent_kinds=None, n_boot=2000):
    """Discrimination = abstain(absent) − abstain(answerable), seed-cluster bootstrap."""
    ab = np.array([p["scores"]["abstain"] for p in per], float)
    isabs = np.array([it["ovm_absent"] for it in items])
    use = np.array([(not it["ovm_absent"]) or absent_kinds is None or it["kind"] in absent_kinds for it in items])
    seeds = np.array([it["seed_raft_idx"] for it in items])
    us = np.unique(seeds[use]); grp = {s: np.where((seeds == s) & use)[0] for s in us}
    f = lambda ii: ab[ii][isabs[ii]].mean() - ab[ii][~isabs[ii]].mean()
    ii0 = np.where(use)[0]
    rng = np.random.default_rng(0)
    bs = [f(np.concatenate([grp[s] for s in rng.choice(us, len(us))])) for _ in range(n_boot)]
    return [float(f(ii0)), *np.percentile(bs, [2.5, 97.5]).tolist()]


def report():
    items = [it for it in _load_items() if it["matched"]]
    out = {"build": json.loads((OUT / "build.json").read_text())}
    v2_reports = {"qa": lambda f: paths.RUNS / "gate2" / f"baseline_qa_iter_val_{f}.json",
                  "andgate": lambda f: paths.RUNS / "gate3b_andgate" / "reports" / f"selected__iter_val_{f}.json",
                  "lora_lr3e-4": lambda f: paths.RUNS / "gate2" / "reports" / f"lora_lr0.0003__iter_val_{f}.json",
                  "fullft": lambda f: paths.RUNS / "gate4_fullft" / "reports" / f"selected__iter_val_{f}.json"}
    gq = {p["id"]: p for p in json.loads((OUT / "gen_qa.json").read_text())}
    for name in MODELS:
        per = json.loads((OUT / f"gen_{name}.json").read_text())
        assert [p["id"] for p in per] == [it["id"] for it in items]
        v2 = []
        for f in (1, 2, 3):
            v2 += [p for p in json.loads(v2_reports[name](f).read_text())["per_item"] if p["kind"] in ANS + ABS]
        v2i = [{"ovm_absent": p["kind"].startswith("abs"), "seed_raft_idx": p["id"].split(":")[0], "kind": p["kind"]} for p in v2]
        row = {"D_matched": _disc_ci(items, per), "D_v2_k12": _disc_ci(v2i, v2),
               "D_matched_del_only": _disc_ci(items, per, DEL), "D_matched_distractor_only": _disc_ci(items, per, ABS)}
        rate = lambda ks: float(np.mean([p["scores"]["abstain"] for p in per if p["kind"] in ks]))
        row["abstain_rate"] = {k: rate((k,)) for k in ANS + ABS + DEL}
        row["matched_answerable_f1"] = float(np.mean([p["scores"]["token_f1"] for p in per if p["kind"] in ANS]))
        if name != "qa":
            row["matched_answerable_f1_delta_vs_qa"] = metrics.paired_delta_ci(
                {p["id"]: p["scores"]["token_f1"] for p in per if p["kind"] in ANS},
                {k: v["scores"]["token_f1"] for k, v in gq.items() if v["kind"] in ANS})
        Dm, Dv = row["D_matched"], row["D_v2_k12"][0]
        row["verdict"] = ("n/a (no abstention)" if Dv < 0.05 else "BEYOND OVERLAP" if Dm[1] >= 0.5 * Dv
                          else "OVERLAP-DOMINATED" if Dm[2] < 0.25 * Dv else "PARTIAL")
        row["ratio_matched_to_v2"] = Dm[0] / Dv if Dv else None
        out[name] = row
    if (OUT / "probe.json").exists():
        out["probe"] = json.loads((OUT / "probe.json").read_text())
    if (OUT / "audit_result.json").exists():
        key = json.loads((OUT / "audit_key.json").read_text()); ans = json.loads((OUT / "audit_result.json").read_text())
        yes = {k: [a["aid"] for a in ans if key[a["aid"]]["kind"] == k and a["answer_in_passage"]] for k in ("del1", "ans1")}
        out["audit"] = {"n_del1": sum(key[a["aid"]]["kind"] == "del1" for a in ans), "n_ans1": sum(key[a["aid"]]["kind"] == "ans1" for a in ans),
                        "contamination_del1_yes": len(yes["del1"]), "sensitivity_ans1_yes": len(yes["ans1"]), "del1_yes_aids": yes["del1"]}
    (OUT / "ovm_numbers.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    {"feasibility": feasibility, "build": build, "eval": eval_models, "probe": probe, "report": report}[sys.argv[1]]()

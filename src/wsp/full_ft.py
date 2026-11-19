"""Full fine-tuning comparison arm (docs/15, pre-registered). Same recipe as the Gate 2 LoRA arm, but
every parameter trains. lr in {1e-5, 3e-5}; selection on iter_val_0 with Gate 2's rule; 3 norm-matched
random controls; held-out iter_val_1..3; full retention + copy probe; numbers pooled vs θ_qa.

    python -m wsp.full_ft run
    python -m wsp.full_ft report
"""
from __future__ import annotations

import json
import math
import random
import shutil
import sys
import time

import numpy as np

from . import checkpoints, config_wsp as C, engine, loop_eval, metrics, paths, probes, retention, splits
from .loop_eval import _by, compare, discrimination

OUT = paths.RUNS / "gate4_fullft"
G2 = paths.RUNS / "gate2"
LRS = (1e-5, 3e-5)


def train_full(lr: float, seed: int = C.SEED, epochs: int = 1, micro: int = 4, accum: int = 4, max_len: int = 1024):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    sys.path.insert(0, str(paths.ROOT / "vendor" / "finetuning"))
    from ft_data import IGNORE, render_custom

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(checkpoints.model_dir("qa"))
    model = AutoModelForCausalLM.from_pretrained(checkpoints.model_dir("qa"), torch_dtype=torch.float32).cuda()
    model.config.eos_token_id = 1
    init = {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items() if k != engine.TIED}
    rows = splits.load("edit_dev")
    random.Random(seed).shuffle(rows)
    data = [render_custom(x["messages"], tok, max_len) for x in rows]
    pad = tok.convert_tokens_to_ids("<|pad|>")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    total = math.ceil(len(data) / (micro * accum)) * epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total), total)
    model.train(); step = 0; t0 = time.time(); losses = []
    for ep in range(epochs):
        for s in range(0, len(data), micro):
            b = data[s:s + micro]
            L = max(len(i) for i, _ in b)
            ids = torch.full((len(b), L), pad); lab = torch.full((len(b), L), IGNORE); att = torch.zeros((len(b), L), dtype=torch.long)
            for j, (i, l) in enumerate(b):
                ids[j, :len(i)] = torch.tensor(i); lab[j, :len(l)] = torch.tensor(l); att[j, :len(i)] = 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda()).loss / accum
            loss.backward(); losses.append(loss.item() * accum)
            if (s // micro + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 100 == 0:
                    print(f"    lr={lr:g} step {step}/{total} loss {np.mean(losses[-100 * accum:]):.4f}", flush=True)
    sd = model.state_dict()
    delta = {k: (sd[k].detach().float().cpu() - init[k]) for k in init}
    info = {"lr": lr, "seed": seed, "epochs": epochs, "steps": step, "n_train": len(data),
            "final_loss_avg_last100": float(np.mean(losses[-100 * accum:])), "wall_s": round(time.time() - t0, 1)}
    del model, opt; torch.cuda.empty_cache()
    return delta, info


def assess(name, delta, base, gate, tok, log, split, ret_ids, baseline):
    m = engine.materialize(base, delta)
    safety = gate.check(delta, base, m)
    log.write("safety", arm=name, split=split, **safety)
    if not safety["pass"]:
        print(f"  [{name}] SAFETY VETO: {safety['vetoes']}", flush=True)
        del m
        return {"arm": name, "split": split, "safety": safety, "vetoed": True}
    rep = loop_eval.evaluate(m, tok, split, ret_ids=ret_ids)
    cmp = compare(rep, baseline)
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    (OUT / "reports" / f"{name}__{split}.json").write_text(json.dumps(rep))
    log.write("eval", arm=name, split=split, agg=rep["agg"], compare=cmp, norms=engine.norms(delta, base)["global_rel"])
    t, dsc = cmp["target_abstain"], cmp["discrimination"]
    print(f"  [{name}] {split}: abstain Δ {t['delta']:+.3f} {[round(x, 3) for x in t['ci']]}  disc k=1/2/3 "
          f"{dsc[1]:+.3f}/{dsc[2]:+.3f}/{dsc[3]:+.3f}  clean F1 Δ {cmp['ans1.token_f1']['delta']:+.3f}  "
          f"guardrails {cmp['guardrails']}", flush=True)
    del m
    return {"arm": name, "split": split, "safety": safety, "compare": cmp, "agg": rep["agg"]}


def run():
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    lock = checkpoints.verify_lock()
    log = engine.RunLog("gate4_fullft")
    log.write("start", lock=lock, lrs=LRS)
    ref, tok, _ = checkpoints.load("qa")
    base = engine.load_state("qa")
    ret = loop_eval.retention_ids()
    canary = torch.from_numpy(np.concatenate([x[:C.CANARY_WINDOWS // 3 + 1] for x in loop_eval.retention_ids(64).values()]).astype(np.int64))
    gate = engine.SafetyGate(ref, canary)
    del ref; torch.cuda.empty_cache()
    bl = {f: json.loads((G2 / f"baseline_qa_iter_val_{f}.json").read_text()) for f in range(4)}
    for f in (1, 2, 3):
        bl[f]["retention"] = bl[0]["retention"]

    cands = {}
    for lr in LRS:
        name = f"fullft_lr{lr:g}"
        p = OUT / "deltas" / f"{name}.safetensors"
        if p.exists():
            delta = engine.load_delta(p)
        else:
            print(f"== training {name}", flush=True)
            delta, info = train_full(lr)
            sha = engine.save_delta(delta, p)
            log.write("train", arm=name, delta_sha256=sha, norms_global=engine.norms(delta, base)["global_rel"], **info)
        cands[name] = (delta, assess(name, delta, base, gate, tok, log, "iter_val_0", ret, bl[0]))
    ok = {n: r for n, (d, r) in cands.items() if not r.get("vetoed") and r["compare"]["guardrails_pass"]}
    valid = {n: r for n, (d, r) in cands.items() if not r.get("vetoed")}
    if ok:
        chosen = max(ok, key=lambda n: ok[n]["compare"]["target_abstain"]["delta"])
    else:
        chosen = max(valid or cands, key=lambda n: cands[n][1].get("compare", {}).get("target_abstain", {}).get("delta", -9))
        print(f"NO full-FT config passed safety + guardrails; continuing with {chosen} for the record (flagged).", flush=True)
    log.write("selection", chosen=chosen, passing=list(ok), flagged=not ok)
    print(f"== selected {chosen}", flush=True)
    sel = cands[chosen][0]
    for n in list(cands):
        if n != chosen:
            cands[n] = None
    shutil.copy(OUT / "deltas" / f"{chosen}.safetensors", OUT / "deltas" / "selected.safetensors")
    (OUT / "selection.json").write_text(json.dumps({"chosen": chosen, "passing": list(ok)}))

    rnd = {}
    for s in range(3):
        d = engine.random_like(sel, seed=s)
        engine.save_delta(d, OUT / "deltas" / f"random_s{s}.safetensors")
        rnd[f"random_s{s}"] = d
        assess(f"random_s{s}", d, base, gate, tok, log, "iter_val_0", ret, bl[0])
    for name, d in [("selected", sel), *rnd.items()]:
        for f in (1, 2, 3):
            assess(name, d, base, gate, tok, log, f"iter_val_{f}", None, bl[f])

    for name, d in [("selected", sel), ("random_s0", rnd["random_s0"])]:
        m = engine.materialize(base, d)
        rr = retention.evaluate(m, OUT / "retention", name)
        pr = probes.run(m, tok)
        (OUT / "probes").mkdir(parents=True, exist_ok=True)
        (OUT / "probes" / f"{name}.json").write_text(json.dumps(pr))
        log.write("milestone", arm=name, retention=rr,
                  probes={k: {kk: vv for kk, vv in v.items() if kk != "per_item"} for k, v in pr.items()})
        print(name, "verbatim copy", pr["C_needle_verbatim"]["rate"], pr["C_needle_verbatim"]["ci95"], flush=True)
        del m; torch.cuda.empty_cache()
    log.write("done")
    print("done", flush=True)


def _pooled(name):
    per = []
    for f in (1, 2, 3):
        p = OUT / "reports" / f"{name}__iter_val_{f}.json" if name != "qa" else G2 / f"baseline_qa_iter_val_{f}.json"
        if not p.exists():
            return None
        per += json.loads(p.read_text())["per_item"]
    return {"per_item": per}


def report():
    base = _pooled("qa")
    out = {"selection": json.loads((OUT / "selection.json").read_text()), "val0": {}, "heldout": {}}
    for r in (json.loads(l) for l in open(OUT / "log.jsonl", encoding="utf-8")):
        if r.get("kind") == "eval" and r["split"] == "iter_val_0":
            c = r["compare"]
            out["val0"][r["arm"]] = {k: c[k] for k in ("target_abstain", "ans1.token_f1", "discrimination", "guardrails", "guardrails_pass")}
            out["val0"][r["arm"]]["global_rel_norm"] = r["norms"]
        if r.get("kind") == "train":
            out.setdefault("train", {})[r["arm"]] = {k: r[k] for k in ("lr", "steps", "final_loss_avg_last100", "wall_s", "norms_global")}
    reps = {}
    for a in ("selected", "random_s0", "random_s1", "random_s2"):
        r = _pooled(a)
        if r is None:
            out["heldout"][a] = "missing (safety veto)"
            continue
        reps[a] = r
        c = compare(r, base)
        out["heldout"][a] = {k: c[k] for k in ("target_abstain", "ans1.token_f1", "ans_all.token_f1", "ans_all.false_abstain",
                                               "ans1.fabrication", "discrimination", "guardrails", "guardrails_pass")}
        out["heldout"][a]["abstain_rate"] = {k: float(np.mean(list(_by(r, k, "abstain").values()))) for k in C.ABS_KINDS + C.ANS_KINDS}
    out["milestone"] = {}
    for r in (json.loads(l) for l in open(OUT / "log.jsonl", encoding="utf-8")):
        if r.get("kind") != "milestone":
            continue
        row = {}
        for s in r["retention"]:
            k = s.replace("/", "__")
            a = np.load(OUT / "retention" / f"{r['arm']}__{k}.npy"); b = np.load(paths.RUNS / "gate1" / "retention" / f"qa__{k}.npy")
            row[s] = {"rel": float(a.mean() / b.mean() - 1), "delta": metrics.paired_delta_ci(dict(enumerate(a)), dict(enumerate(b)))}
        row["retention_pass"] = all(v["rel"] <= C.RETENTION_MAX_REL_INCREASE for v in row.values() if isinstance(v, dict))
        row["verbatim_copy"] = {kk: vv for kk, vv in r["probes"]["C_needle_verbatim"].items() if not isinstance(vv, (list, dict))}
        out["milestone"][r["arm"]] = row
    (OUT / "gate4_fullft_numbers.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()

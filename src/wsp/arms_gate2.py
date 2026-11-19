"""Gate 2 arms — no editor LLM yet. Pre-registered before running:

  S2 LoRA     : r=16, alpha=32, dropout 0.05, all 7 linear projections (q,k,v,o,gate,up,down),
                trained on edit_dev v2 (six item kinds, document count decorrelated -> 1/2 abstention targets), 1 epoch,
                micro 8 x accum 2 (eff. batch 16), AdamW betas (0.9,0.95), wd 0, cosine with 3%
                warmup, grad clip 1.0, bf16 autocast, seed 1337, loss on assistant tokens only
                (vendor ft_data.render_custom — θ_qa's own training format).
                Two learning rates {1e-4, 3e-4}; SELECTION on iter_val_0 only: among configs that
                pass the safety gate AND every approved guardrail, pick the largest abstention gain.
  S1 Random   : for the selected LoRA delta, 3 seeds of random_like(): same tensors, same rank,
                same per-tensor Frobenius norm; only the direction differs.
  Report      : selected LoRA and the random seeds on iter_val_1..3 pooled (never used for any
                choice), paired vs θ_qa on the same items; full held-out retention + Phase-0
                copy probe for the selected LoRA and random seed 0.

    python -m wsp.arms_gate2
"""
from __future__ import annotations

import json
import math
import random
import sys
import time

import numpy as np

from . import checkpoints, config_wsp as C, engine, loop_eval, paths, splits

OUT = paths.RUNS / "gate2"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LRS = (1e-4, 3e-4)


def train_lora(lr: float, seed: int = C.SEED, r: int = 16, alpha: int = 32, epochs: int = 1,
               micro: int = 8, accum: int = 2, max_len: int = 1024) -> tuple[dict, dict]:
    import torch
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import LoraLayer
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    sys.path.insert(0, str(paths.ROOT / "vendor" / "finetuning"))
    from ft_data import IGNORE, render_custom  # θ_qa's own training renderer (read-only import)

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(checkpoints.model_dir("qa"))
    model = AutoModelForCausalLM.from_pretrained(checkpoints.model_dir("qa"), torch_dtype=torch.float32).cuda()
    model.config.eos_token_id = 1
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05,
                                             target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM"))
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
                if step % 50 == 0:
                    print(f"    lr={lr:g} step {step}/{total} loss {np.mean(losses[-50 * accum:]):.4f}", flush=True)
    delta = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoraLayer):
            key = name.replace("base_model.model.", "") + ".weight"
            delta[key] = mod.get_delta_weight("default").detach().float().cpu()
    info = {"lr": lr, "seed": seed, "r": r, "alpha": alpha, "epochs": epochs, "steps": step,
            "n_train": len(data), "final_loss_avg_last50": float(np.mean(losses[-50 * accum:])),
            "wall_s": round(time.time() - t0, 1)}
    del model; torch.cuda.empty_cache()
    return delta, info


def assess(name, delta, base, gate, tok, log, split, ret_ids, baseline):
    m = engine.materialize(base, delta)
    safety = gate.check(delta, base, m)
    log.write("safety", arm=name, split=split, **{k: v for k, v in safety.items()})
    if not safety["pass"]:
        print(f"  [{name}] SAFETY VETO: {safety['vetoes']}", flush=True)
        return {"arm": name, "split": split, "safety": safety, "vetoed": True}
    rep = loop_eval.evaluate(m, tok, split, ret_ids=ret_ids)
    cmp = loop_eval.compare(rep, baseline)
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    (OUT / "reports" / f"{name}__{split}.json").write_text(json.dumps(rep))
    log.write("eval", arm=name, split=split, agg=rep["agg"], compare=cmp, norms=engine.norms(delta, base)["global_rel"])
    t, dsc = cmp["target_abstain"], cmp["discrimination"]
    print(f"  [{name}] {split}: abstain Δ {t['delta']:+.3f} {[round(x, 3) for x in t['ci']]}  discrimination k=1/2/3 "
          f"{dsc[1]:+.3f}/{dsc[2]:+.3f}/{dsc[3]:+.3f}  clean F1 Δ {cmp['ans1.token_f1']['delta']:+.3f}  "
          f"false-abst(ans) Δ {cmp['ans_all.false_abstain']['delta']:+.3f}  guardrails {cmp['guardrails']}", flush=True)
    del m
    return {"arm": name, "split": split, "safety": safety, "compare": cmp, "agg": rep["agg"]}


def main():
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    lock = checkpoints.verify_lock()
    log = engine.RunLog("gate2")
    log.write("start", lock=lock, config={k: getattr(C, k) for k in dir(C) if k.isupper()}, lrs=LRS,
              pool_manifest=json.loads((paths.SPLITS / "item_pool_manifest.json").read_text())["pool_sha256"])
    ref, tok, _ = checkpoints.load("qa")
    base = engine.load_state("qa")
    ret = loop_eval.retention_ids()
    canary = torch.from_numpy(np.concatenate([x[:C.CANARY_WINDOWS // 3 + 1] for x in loop_eval.retention_ids(64).values()]).astype(np.int64))
    gate = engine.SafetyGate(ref, canary)
    del ref; torch.cuda.empty_cache()
    bl = {f: json.loads((OUT / f"baseline_qa_iter_val_{f}.json").read_text()) for f in range(4)}
    for f in (1, 2, 3):                                   # retention is fold-independent
        bl[f]["retention"] = bl[0]["retention"]

    # ---- selection on iter_val_0 ----
    cands = {}
    for lr in LRS:
        name = f"lora_lr{lr:g}"
        print(f"== training {name}", flush=True)
        delta, info = train_lora(lr)
        sha = engine.save_delta(delta, OUT / "deltas" / f"{name}.safetensors")
        log.write("train", arm=name, delta_sha256=sha, **info)
        cands[name] = (delta, assess(name, delta, base, gate, tok, log, "iter_val_0", ret, bl[0]))
    ok = {n: r for n, (d, r) in cands.items() if not r.get("vetoed") and r["compare"]["guardrails_pass"]}
    if not ok:
        log.write("selection", chosen=None, reason="no config passed safety + guardrails on iter_val_0")
        print("NO LoRA CONFIG PASSED — stop and report.", flush=True)
        chosen = max(cands, key=lambda n: cands[n][1].get("compare", {}).get("target_abstain", {}).get("delta", -9))
        print(f"(continuing with best-target config {chosen} for the record, flagged as guardrail-failing)", flush=True)
    else:
        chosen = max(ok, key=lambda n: ok[n]["compare"]["target_abstain"]["delta"])
    log.write("selection", chosen=chosen, rule="max abstain gain among safety+guardrail passing configs on iter_val_0",
              passing=list(ok))
    print(f"== selected {chosen}", flush=True)
    lora = cands[chosen][0]

    # ---- random controls (rank- and norm-matched) ----
    rnd = {}
    for s in range(3):
        d = engine.random_like(lora, seed=s)
        engine.save_delta(d, OUT / "deltas" / f"random_s{s}.safetensors")
        rnd[f"random_s{s}"] = d
        assess(f"random_s{s}", d, base, gate, tok, log, "iter_val_0", ret, bl[0])

    # ---- held-out report folds 1..3 ----
    for name, d in [(chosen, lora), *rnd.items()]:
        for f in (1, 2, 3):
            assess(name, d, base, gate, tok, log, f"iter_val_{f}", None, bl[f])
    log.write("done")
    print("done", flush=True)


if __name__ == "__main__":
    main()

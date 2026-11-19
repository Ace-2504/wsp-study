"""Gate 3a — deterministic weight-space search over the saved Gate 2 LoRA deltas (no LLM, no training).
Pre-registered in docs/08_GATE3_PLAN.md:

  A  scale sweep      θ_qa + s·Δ, s in SCALES, for both LoRA deltas
  B  attribution      Δ_lr3e-4 restricted to {attn, mlp} x {L0-3, L4-7, L8-11}, scale 1.0
  C  efficient subset union of the top-k (k=1..3) groups by efficiency, scales {0.25, 0.5, 1.0}
  Selection on iter_val_0: largest abstention gain among configs passing safety + every approved
  guardrail + validity (discrimination min >= half the gain). None -> stop, report the frontier.
  Selected edit -> iter_val_1..3 pooled, 3 matched random controls, full retention + probes.

    python -m wsp.gate3_sweep            # resumable: configs already in results.json are skipped
"""
from __future__ import annotations

import json

import numpy as np

from . import checkpoints, config_wsp as C, engine, loop_eval, paths, probes, retention

OUT = paths.RUNS / "gate3a"
SRC = paths.RUNS / "gate2" / "deltas"
SCALES = (0.05, 0.1, 0.2, 0.3, 0.5)
C_SCALES = (0.25, 0.5, 1.0)
BANDS = {"L0-3": range(0, 4), "L4-7": range(4, 8), "L8-11": range(8, 12)}
MODS = {"attn": ".self_attn.", "mlp": ".mlp."}
GROUPS = [(m, b) for m in MODS for b in BANDS]


def group_mask(delta: dict, groups) -> dict:
    keep = {}
    for k, v in delta.items():
        layer = int(k.split(".")[2])
        if any(MODS[m] in k and layer in BANDS[b] for m, b in groups):
            keep[k] = v
    return keep


def scaled(delta: dict, s: float) -> dict:
    return {k: s * v for k, v in delta.items()}


def valid(cmp: dict) -> bool:
    g, dmin = cmp["target_abstain"]["delta"], cmp["discrimination"]["min"]
    return g > 0 and dmin is not None and dmin >= 0.5 * g


class Runner:
    def __init__(self, out=None, exp: str = "gate3a"):
        import torch
        self.out = out or OUT
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = checkpoints.verify_lock()
        self.log = engine.RunLog(exp)
        ref, self.tok, _ = checkpoints.load("qa")
        self.base = engine.load_state("qa")
        self.ret = loop_eval.retention_ids()
        canary = torch.from_numpy(np.concatenate([x[:C.CANARY_WINDOWS // 3 + 1]
                                                  for x in loop_eval.retention_ids(64).values()]).astype(np.int64))
        self.gate = engine.SafetyGate(ref, canary)
        del ref; torch.cuda.empty_cache()
        G2 = paths.RUNS / "gate2"
        self.bl = {f: json.loads((G2 / f"baseline_qa_iter_val_{f}.json").read_text()) for f in range(4)}
        for f in (1, 2, 3):
            self.bl[f]["retention"] = self.bl[0]["retention"]
        self.rp = self.out / "results.json"
        self.res = json.loads(self.rp.read_text()) if self.rp.exists() else {}

    def run(self, name: str, delta: dict, split: str = "iter_val_0", meta: dict | None = None) -> dict:
        key = f"{name}__{split}"
        if key in self.res:
            return self.res[key]
        m = engine.materialize(self.base, delta)
        safety = self.gate.check(delta, self.base, m)
        self.log.write("safety", arm=name, split=split, meta=meta, **safety)
        rec = {"arm": name, "split": split, "meta": meta or {}, "safety": {k: v for k, v in safety.items()},
               "norm": engine.norms(delta, self.base)["global_rel"]}
        if not safety["pass"]:
            rec["vetoed"] = True
            print(f"  [{name}] SAFETY VETO {safety['vetoes']}", flush=True)
        else:
            f = int(split[-1])
            rep = loop_eval.evaluate(m, self.tok, split, ret_ids=self.ret if f == 0 else None)
            cmp = loop_eval.compare(rep, self.bl[f])
            (self.out / "reports").mkdir(parents=True, exist_ok=True)
            (self.out / "reports" / f"{key}.json").write_text(json.dumps(rep))
            rec["compare"] = cmp
            rec["valid"] = valid(cmp)
            rec["selectable"] = bool(cmp["guardrails_pass"] and rec["valid"])
            self.log.write("eval", arm=name, split=split, meta=meta, compare=cmp, norm=rec["norm"], valid=rec["valid"])
            t, dsc = cmp["target_abstain"], cmp["discrimination"]
            rt = cmp.get("retention", {})
            print(f"  [{name}] {split} |Δ|={rec['norm']:.4f} abstain Δ {t['delta']:+.3f} disc.min {dsc['min']:+.3f} "
                  f"cleanF1 Δ {cmp['ans1.token_f1']['delta']:+.3f} false-abst Δ {cmp['ans_all.false_abstain']['delta']:+.3f} "
                  f"ret " + "/".join(f"{100 * v['rel']:+.2f}%" for v in rt.values()) +
                  f" guard {cmp['guardrails']} valid {rec['valid']}", flush=True)
        del m
        self.res[key] = rec
        self.rp.write_text(json.dumps(self.res, default=float))
        return rec


def mean_ret(rec):
    return float(np.mean([v["rel"] for v in rec["compare"]["retention"].values()]))


def main():
    R = Runner()
    R.log.write("start", lock=R.lock, scales=SCALES, c_scales=C_SCALES, groups=GROUPS,
                config={k: getattr(C, k) for k in dir(C) if k.isupper()})
    src = {lr: engine.load_delta(SRC / f"lora_lr{lr}.safetensors") for lr in ("0.0001", "0.0003")}

    print("== Phase A: scale sweep", flush=True)
    for lr, d in src.items():
        for s in SCALES:
            R.run(f"A_lr{lr}_s{s:g}", scaled(d, s), meta={"phase": "A", "lr": lr, "scale": s})

    print("== Phase B: attribution (lr3e-4, scale 1.0)", flush=True)
    B = {}
    for m, b in GROUPS:
        B[(m, b)] = R.run(f"B_{m}_{b}", group_mask(src["0.0003"], [(m, b)]),
                          meta={"phase": "B", "groups": [[m, b]], "scale": 1.0})

    eff = []
    for g, rec in B.items():
        if rec.get("vetoed"):
            continue
        t = rec["compare"]["target_abstain"]
        if t["significant"] and t["delta"] > 0:
            eff.append((t["delta"] / max(mean_ret(rec), 1e-3), g))
    eff.sort(reverse=True)
    R.log.write("phaseB_ranking", ranking=[{"group": list(g), "efficiency": e} for e, g in eff])
    print("efficiency ranking:", [(f"{g[0]}_{g[1]}", round(e, 2)) for e, g in eff], flush=True)

    print("== Phase C: efficient subsets", flush=True)
    for k in (1, 2, 3):
        if len(eff) < k:
            break
        gs = [g for _, g in eff[:k]]
        for s in C_SCALES:
            if k == 1 and s == 1.0:
                continue                                   # identical to its Phase B config
            tag = "+".join(f"{m}_{b}" for m, b in gs)
            R.run(f"C_top{k}_{tag}_s{s:g}", scaled(group_mask(src["0.0003"], gs), s),
                  meta={"phase": "C", "groups": [list(g) for g in gs], "scale": s, "k": k})

    dev = {k: v for k, v in R.res.items() if k.endswith("__iter_val_0") and not v.get("vetoed")}
    ok = {k: v for k, v in dev.items() if v["selectable"]}
    if not ok:
        R.log.write("selection", chosen=None, reason="no config passed safety + guardrails + validity on iter_val_0")
        print("NO CONFIG PASSED — Gate 3a stops here; report the frontier.", flush=True)
        return
    ck = max(ok, key=lambda k: ok[k]["compare"]["target_abstain"]["delta"])
    chosen = ok[ck]
    R.log.write("selection", chosen=chosen["arm"], meta=chosen["meta"], passing=[v["arm"] for v in ok.values()])
    print(f"== selected {chosen['arm']}", flush=True)

    # rebuild the chosen delta from its meta
    mt = chosen["meta"]
    if mt["phase"] == "A":
        delta = scaled(src[mt["lr"]], mt["scale"])
    else:
        delta = scaled(group_mask(src["0.0003"], [tuple(g) for g in mt["groups"]]), mt["scale"])
    engine.save_delta(delta, OUT / "deltas" / "selected.safetensors")
    arms = {"selected": delta}
    for s in range(3):
        arms[f"random_s{s}"] = engine.random_like(delta, seed=s)
        engine.save_delta(arms[f"random_s{s}"], OUT / "deltas" / f"random_s{s}.safetensors")
        R.run(f"random_s{s}", arms[f"random_s{s}"], "iter_val_0", meta={"control_of": chosen["arm"]})
    for name, d in arms.items():
        for f in (1, 2, 3):
            R.run(name, d, f"iter_val_{f}", meta={"selected": chosen["arm"]} if name == "selected" else {"control_of": chosen["arm"]})

    print("== milestone: full retention + probes", flush=True)
    for name in ("selected", "random_s0"):
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
    main()

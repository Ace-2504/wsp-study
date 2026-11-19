"""Held-out scoring of each editor run's nominated edit (docs/16). Experimenter only.

    python -m wsp.editor_report score     # GPU: held-out folds, random control, retention+copy, held-out matched pool
    python -m wsp.editor_report report    # numbers -> gate5_editor_numbers.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import batchgen, checkpoints, config_wsp as C, engine, metrics, paths, probes, retention
from .editor_step import OUT, build_delta, _steps, _gpu_lock
from .loop_eval import _by, compare
from . import ovm_pool as O

RUNS = ("A", "B", "C")
AG_D = 0.122          # docs/14 AND-gate held-out matched D (point), the pre-registered bar


def _final(r):
    p = OUT / f"run_{r}" / "final.json"
    return json.loads(p.read_text()) if p.exists() else None


def score(runs=RUNS):
    import torch
    from .gate3_sweep import Runner
    items = [it for it in O._load_items() if it["matched"]]
    for r in runs:
        fin = _final(r)
        if fin is None:
            print(f"run {r}: no nomination", flush=True)
            continue
        R = Runner(out=OUT / f"run_{r}", exp=f"gate5_editor/run_{r}")
        delta = build_delta(fin["action"], R.base)
        engine.save_delta(delta, OUT / f"run_{r}" / "final.safetensors")
        rnd = engine.random_like(delta, seed=0)
        for name, d in (("final", delta), ("random_s0", rnd)):
            for f in (1, 2, 3):
                if (OUT / f"run_{r}" / "reports" / f"{name}__iter_val_{f}.json").exists():
                    continue
                with _gpu_lock():
                    R.run(name, d, f"iter_val_{f}", meta=fin["action"] if name == "final" else {"control": "random_s0"})
        for name, d in (("final", delta), ("random_s0", rnd)):
            if (OUT / f"run_{r}" / "probes" / f"{name}.json").exists():
                continue
            with _gpu_lock():
                m = engine.materialize(R.base, d)
                rr = retention.evaluate(m, OUT / f"run_{r}" / "retention", name)
                pr = probes.run(m, R.tok)
                del m; torch.cuda.empty_cache()
            (OUT / f"run_{r}" / "probes").mkdir(parents=True, exist_ok=True)
            (OUT / f"run_{r}" / "probes" / f"{name}.json").write_text(json.dumps({"retention": rr, "copy": {k: v for k, v in pr["C_needle_verbatim"].items() if k != "per_item"}}, default=float))
        gp = OUT / f"run_{r}" / "heldout_ovm_gen.json"
        if not gp.exists():
            with _gpu_lock():
                _, tok, eos = checkpoints.load("qa")
                m = engine.materialize(R.base, delta)
                resp = batchgen.generate(m, tok, eos, [(i["messages"][0]["content"], i["messages"][1]["content"]) for i in items], C.GEN_BATCH)
                del m; torch.cuda.empty_cache()
            per = [{"id": it["id"], "kind": it["kind"], "seed": it["seed_raft_idx"], "resp": x,
                    "scores": metrics.score_item(x, it["messages"][2]["content"], it["messages"][1]["content"], not it["ovm_absent"])}
                   for it, x in zip(items, resp)]
            gp.write_text(json.dumps(per))
        print(f"run {r} scored", flush=True)


def _pooled(r, name):
    per = []
    for f in (1, 2, 3):
        p = (OUT / f"run_{r}" / "reports" / f"{name}__iter_val_{f}.json") if r else (paths.RUNS / "gate2" / f"baseline_qa_iter_val_{f}.json")
        per += json.loads(p.read_text())["per_item"]
    return {"per_item": per}


def report(runs=RUNS, fname="gate5_editor_numbers.json"):
    base = _pooled(None, "qa")
    items = [it for it in O._load_items() if it["matched"]]
    gq = {p["id"]: p for p in json.loads((O.OUT / "gen_qa.json").read_text())}
    out = {}
    for r in runs:
        fin = _final(r)
        st = _steps(OUT / f"run_{r}")
        row = {"n_steps": len(st), "final": fin,
               "trajectory": [{"step": s["step"], "components": s["action"]["components"], "note": s["action"].get("note"),
                               "veto": not s["safety"]["pass"],
                               "abst_gain": s.get("dev", {}).get("abstain_gain_abs"), "guard": s.get("dev", {}).get("guardrails_pass"),
                               "ret": s.get("dev", {}).get("retention_pct"), "ovm_D": s.get("ovm", {}).get("D_matched")} for s in st]}
        if fin is None:
            row["outcome"] = "WORSE (no nomination)"
            out[r] = row
            continue
        for name in ("final", "random_s0"):
            rep = _pooled(r, name)
            c = compare(rep, base)
            row[name] = {k: c[k] for k in ("target_abstain", "ans1.token_f1", "ans_all.false_abstain", "ans1.fabrication",
                                           "discrimination", "guardrails", "guardrails_pass")}
            row[name]["abstain_rate"] = {k: float(np.mean(list(_by(rep, k, "abstain").values()))) for k in C.ABS_KINDS + C.ANS_KINDS}
            pr = json.loads((OUT / f"run_{r}" / "probes" / f"{name}.json").read_text())
            ret = {}
            for s in pr["retention"]:
                k = s.replace("/", "__")
                a = np.load(OUT / f"run_{r}" / "retention" / f"{name}__{k}.npy"); b = np.load(paths.RUNS / "gate1" / "retention" / f"qa__{k}.npy")
                ret[s] = float(a.mean() / b.mean() - 1)
            row[name]["retention_rel"] = ret
            row[name]["copy"] = pr["copy"]
        per = json.loads((OUT / f"run_{r}" / "heldout_ovm_gen.json").read_text())
        Dm = O._disc_ci(items, per)
        v2 = [p for p in _pooled(r, "final")["per_item"] if p["kind"] in O.ANS + O.ABS]
        v2i = [{"ovm_absent": p["kind"].startswith("abs"), "seed_raft_idx": p["id"].split(":")[0], "kind": p["kind"]} for p in v2]
        Dv = O._disc_ci(v2i, v2)
        row["ovm"] = {"D_matched": Dm, "D_v2_k12": Dv, "D_del_only": O._disc_ci(items, per, O.DEL),
                      "abstain_rate": {k: float(np.mean([p["scores"]["abstain"] for p in per if p["kind"] == k])) for k in O.ANS + O.ABS + O.DEL},
                      "f1_delta_vs_qa": metrics.paired_delta_ci({p["id"]: p["scores"]["token_f1"] for p in per if p["kind"] in O.ANS},
                                                               {k: v["scores"]["token_f1"] for k, v in gq.items() if v["kind"] in O.ANS})}
        row["ovm"]["verdict"] = ("n/a" if Dv[0] < 0.05 else "BEYOND OVERLAP" if Dm[1] >= 0.5 * Dv[0]
                                 else "OVERLAP-DOMINATED" if Dm[2] < 0.25 * Dv[0] else "PARTIAL")
        g = row["final"]["guardrails_pass"] and all(v <= C.RETENTION_MAX_REL_INCREASE for v in row["final"]["retention_rel"].values()) \
            and row["final"]["copy"]["rate"] >= C.VERBATIM_COPY_MIN
        row["all_guardrails_heldout"] = bool(g)
        row["outcome"] = ("WORSE" if not g else "BEATS DETERMINISTIC" if Dm[1] > AG_D else "MATCHES" if Dm[2] >= AG_D else "WORSE")
        out[r] = row
    n_beats = sum(out[r]["outcome"] == "BEATS DETERMINISTIC" for r in runs)
    if fname == "gate5_random_numbers.json":   # docs/17 sequential rule
        if runs[0] == "R1" and out["R1"]["outcome"] == "BEATS DETERMINISTIC":
            out["arm_verdict"] = "MENU SUFFICES (R1 beats; sequential stop, R2-R3 not run)"
        else:
            out["arm_verdict"] = ("RANDOM MATCHES EDITORS" if n_beats >= 2
                                  else f"EDITOR CHOICES ADD VALUE ({n_beats}/{len(runs)} random runs beat)")
    else:
        out["arm_verdict"] = "EDITOR BEATS DETERMINISTIC" if n_beats >= 2 else f"NOT BETTER ({n_beats}/{len(runs)} runs beat)"
    (OUT / fname).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score(tuple(sys.argv[2:]) or RUNS)
    elif sys.argv[2:]:
        report(tuple(sys.argv[2:]), "gate5_random_numbers.json")
    else:
        report()

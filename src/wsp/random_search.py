"""Same-menu random-search control (docs/17, pre-registered). Uniform random actions over the editors'
action space, 18 evaluations through the same gateway, mechanical nomination by the editors' objective.

    python -m wsp.random_search draw R1      # print the 18 actions (no GPU)
    python -m wsp.random_search run R1       # evaluate them (GPU, resumable) and nominate
"""
from __future__ import annotations

import json
import random
import sys

from . import engine
from .editor_step import BUDGET, MODULES, OUT, SOURCES, _gpu_lock, _steps, build_delta, evaluate

SEEDS = {"R1": 1, "R2": 2, "R3": 3}


def _component(rng):
    if rng.random() < 0.5:
        return {"type": "and_gate", "layer": rng.choice([5, 6]), "rho": rng.choice([0.25, 0.5, 1]),
                "tau": rng.choice([0, 1]), "gain": round(rng.uniform(0, 2), 2)}
    pairs = [(a, b) for a in range(12) for b in range(a, 12)]
    a, b = rng.choice(pairs)
    while True:
        mods = [m for m in MODULES if rng.random() < 0.5]
        if mods:
            break
    return {"type": "scaled_delta", "source": rng.choice(list(SOURCES)), "scale": round(rng.uniform(0, 1), 2),
            "layers": list(range(a, b + 1)), "modules": mods}


def draw(run):
    rng = random.Random(SEEDS[run])
    base = engine.load_state("qa")
    acts, redraws = [], 0
    while len(acts) < BUDGET:
        comps = [_component(rng) for _ in range(rng.choice([1, 2, 3]))]
        gl = [c["layer"] for c in comps if c["type"] == "and_gate"]
        act = {"components": comps, "note": f"random search {run}, seed {SEEDS[run]}"}
        if len(gl) != len(set(gl)):
            redraws += 1
            continue
        try:
            build_delta(act, base)
        except (ValueError, KeyError, TypeError):
            redraws += 1
            continue
        acts.append(act)
    return acts, redraws


def run(r):
    acts, redraws = draw(r)
    d = OUT / f"run_{r}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "drawn_actions.json").write_text(json.dumps({"seed": SEEDS[r], "redraws": redraws, "actions": acts}, indent=1))
    for i in range(len(_steps(d)), BUDGET):
        with _gpu_lock():
            o = evaluate(acts[i], d)
        dv, ov = o.get("dev", {}), o.get("ovm", {})
        print(f"[{r}] step {o['step']} safety {o['safety']['pass']} pass {dv.get('guardrails_pass')} "
              f"gain {dv.get('abstain_gain_abs')} ret {dv.get('retention_pct')} D {ov.get('D_matched')}", flush=True)
    st = _steps(d)
    ok = [s for s in st if s["safety"]["pass"] and s.get("dev", {}).get("guardrails_pass")]
    if not ok:
        (d / "final.json").unlink(missing_ok=True)
        print(f"[{r}] no step passes every dev guardrail -> no nomination", flush=True)
        return
    best = max(ok, key=lambda s: (s["ovm"]["D_matched"][0], s["dev"]["abstain_gain_abs"]))
    (d / "final.json").write_text(json.dumps({"final_step": best["step"], "action": best["action"]}))
    print(f"[{r}] {len(ok)}/{len(st)} pass; final = step {best['step']} D {best['ovm']['D_matched']}", flush=True)


if __name__ == "__main__":
    cmd, r = sys.argv[1], sys.argv[2]
    if cmd == "draw":
        a, n = draw(r)
        print(json.dumps(a, indent=1)); print("redraws", n)
    else:
        run(r)

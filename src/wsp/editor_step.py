"""Editor-LLM arm gateway (docs/16, pre-registered). The ONLY interface an editor agent uses.

It applies a JSON edit to a copy of θ_qa and returns DEV metrics only (iter_val_0 + fast retention +
an overlap-matched pool built from iter_val_0 seeds). It never reads or prints held-out folds.
Every call is logged; the budget is enforced per run.

    python -m wsp.editor_step --run A --brief
    python -m wsp.editor_step --run A --action '{"components": [...], "note": "why"}'
    python -m wsp.editor_step --run A --history
    python -m wsp.editor_step --run A --final 7
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from . import config_wsp as C, engine, metrics, paths

OUT = paths.RUNS / "gate5_editor"
BUDGET = 18
SOURCES = {"lora_lr0.0001": paths.RUNS / "gate2" / "deltas" / "lora_lr0.0001.safetensors",
           "lora_lr0.0003": paths.RUNS / "gate2" / "deltas" / "lora_lr0.0003.safetensors",
           "fullft_lr1e-05": paths.RUNS / "gate4_fullft" / "deltas" / "fullft_lr1e-05.safetensors"}
MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed", "norm")

BRIEF = f"""EDITOR BRIEF — weight edits to a 125.8M LLaMA legal QA model (θ_qa: 12 layers, hidden 768, MLP 3072, SwiGLU).

GOAL. θ_qa almost never abstains. Make it reply with the exact abstain string when the answer is NOT in the
provided documents, while still answering when it is. The documents for absent items are either topical
distractors (abs1/abs2/abs3: 1-3 docs, none containing the answer) or, in the overlap-matched pool, the golden
passage with the answer sentence DELETED (del1/del2). Items with the answer are ans1/ans2/ans3.
IMPORTANT: a word-overlap rule ("docs don't share the question's words") already separates abs from ans at
AUROC 0.97, but is at chance (0.50) on the overlap-matched pool. An edit that only learns overlap will show
large abs-vs-ans discrimination but near-zero discrimination on the matched pool (`ovm.D_matched`).

HARD GUARDRAILS (an edit that breaks any is unusable):
  - retention: held-out LM loss rise <= +0.5% in EVERY domain (sec, case-law, web) vs θ_qa;
  - clean F1 (ans1) paired delta: lower 95% CI >= -0.02;  fabrication must not rise significantly;
  - safety gate (automatic veto): per-tensor ||dW||/||W|| <= 0.25, global <= 0.05, activation ratio <= 2,
    canary loss <= +10%.
OBJECTIVE for your final pick: among edits passing all guardrails on dev, maximise discrimination on the
matched pool (ovm.D_matched) first, and abstention gain on abs items second.

ACTION SPACE. An action is JSON: {{"components": [ ... up to 3 ... ], "note": "<your reasoning, 1-3 sentences>"}}.
Components are summed into one weight delta applied to θ_qa:
 1. {{"type": "and_gate", "layer": 5|6, "rho": 0.25|0.5|1, "tau": 0|1, "gain": 0..2}}
    Repurposes ONE low-importance MLP neuron in that layer as an AND-gate:
      gate row = S*(w - tau*c)  (w: linear answer-absence probe on the MLP input, fit on training data;
                                 c: near-constant direction, so tau raises the firing threshold),
      up row   = r              (a detector that is ~1 only at the first reply token, ~0 inside ordinary text),
      down col = gain * delta_rho / m   (delta_rho: write vector optimised to produce the abstain string,
                                 norm-bounded by rho; m: median firing activation).
    At most one and_gate per layer.
 2. {{"type": "scaled_delta", "source": one of {list(SOURCES)}, "scale": 0..1,
     "layers": [subset of 0..11] (default all), "modules": [subset of {list(MODULES)}] (default all)}}
    Adds scale x (a mask of) an existing gradient-trained delta. lora_* are rank-16 LoRA deltas on all 7
    projections; fullft_* is a full fine-tune (every tensor). At scale 1 all three install strong abstention
    (~0.9) but cost +4% to +9% retention (far over budget); "embed"/"norm" exist only in fullft.

EACH CALL costs 1 of your {BUDGET} evaluations (vetoes included) and takes ~3-5 min. It returns, on the DEV fold:
  abstain rate per kind, abstain gain, discrimination per doc count, clean-F1 delta (+CI), false-abstain delta,
  fabrication delta, retention % per domain, guardrail flags, safety, ||dθ||/||θ||, matched-pool D (+CI) and
  abstain on del1/del2, plus a few example failures. θ_qa itself: abstain ~0.00 everywhere, clean F1 ~0.24.

Finish by nominating ONE step with --final <step>. It should pass every guardrail on dev. If none does,
nominate nothing and say so.
"""


def _dev_ovm():
    """Overlap-matched pool from iter_val_0 seeds (dev fold), built exactly as docs/14 but on dev."""
    p = OUT / "dev_ovm_items.jsonl"
    if p.exists():
        return [json.loads(l) for l in open(p, encoding="utf-8")]
    from . import ovm_pool as O
    items, drops, n = O.make_items(["iter_val_0"])
    sc = O.lexical_model()(items)
    keep = set(O.match(items, sc))
    items = [it for i, it in enumerate(items) if i in keep]
    au = O.cluster_auroc(items, sc[sorted(keep)])
    OUT.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    (OUT / "dev_ovm_build.json").write_text(json.dumps({"n_seeds": n, "drops": drops, "n_matched": len(items),
                                                         "lexical_auroc": au}, default=float))
    return items


def build_delta(action, base):
    import torch
    from .and_gate_edit import OUT as AG
    from .neuron_edit import OUT as NE
    comps = action.get("components")
    if not isinstance(comps, list) or not 1 <= len(comps) <= 3:
        raise ValueError("components must be a list of 1-3 items")
    delta, ag_layers = {}, set()
    for c in comps:
        t = c.get("type")
        if t == "and_gate":
            l, rho, tau, gain = int(c["layer"]), float(c["rho"]), float(c["tau"]), float(c.get("gain", 1.0))
            if l not in (5, 6) or rho not in (0.25, 0.5, 1.0) or tau not in (0.0, 1.0) or not 0 <= gain <= 2:
                raise ValueError(f"and_gate out of range: {c}")
            if l in ag_layers:
                raise ValueError("at most one and_gate per layer")
            ag_layers.add(l)
            v = np.load(NE / "vectors" / f"L{l}.npz")
            meta = json.loads((AG / "vectors" / "build_meta.json").read_text())["layers"][str(l)]
            j, S = meta["neuron"], meta["s"]
            m = meta["probe_real"][f"tau{tau:g}"]["m"]
            r = torch.tensor(np.load(AG / "vectors" / f"L{l}.npz")["r"])
            kg, ku, kd = (f"model.layers.{l}.mlp.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj"))
            d = {k: torch.zeros_like(base[k]) for k in (kg, ku, kd)}
            d[kg][j] = S * (torch.tensor(v["w_real"]) - tau * torch.tensor(v["c"])) - base[kg][j]
            d[ku][j] = r - base[ku][j]
            d[kd][:, j] = gain * torch.tensor(v[f"delta_rho{rho:g}"]) / m - base[kd][:, j]
        elif t == "scaled_delta":
            src, scale = c["source"], float(c["scale"])
            if src not in SOURCES or not 0 <= scale <= 1:
                raise ValueError(f"scaled_delta out of range: {c}")
            layers = set(int(x) for x in c.get("layers", range(12)))
            mods = set(c.get("modules", MODULES))
            if not layers <= set(range(12)) or not mods <= set(MODULES):
                raise ValueError(f"bad layers/modules: {c}")
            d = {}
            for k, x in engine.load_delta(SOURCES[src]).items():
                mod = ("embed" if "embed_tokens" in k else "norm" if "norm" in k else
                       next((mm for mm in MODULES[:7] if f".{mm}." in k), None))
                lay = int(k.split("layers.")[1].split(".")[0]) if "layers." in k else None
                if mod in mods and (lay is None or lay in layers):
                    d[k] = scale * x.float()
        else:
            raise ValueError(f"unknown component type {t!r}")
        for k, x in d.items():
            delta[k] = delta[k] + x if k in delta else x
    return delta


def _rate(per, kinds):
    v = [p["scores"]["abstain"] for p in per if p["kind"] in kinds]
    return round(float(np.mean(v)), 3) if v else None


def evaluate(action, run_dir):
    import torch
    from . import batchgen, checkpoints
    from .gate3_sweep import Runner
    from .ovm_pool import _disc_ci
    R = Runner(out=run_dir, exp=f"gate5_editor/{run_dir.name}")
    delta = build_delta(action, R.base)
    step = len(_steps(run_dir)) + 1
    rec = R.run(f"step{step}", delta, "iter_val_0", meta=action)
    out = {"step": step, "action": action, "norm_global_rel": round(rec["norm"], 5),
           "safety": {"pass": rec["safety"]["pass"], "vetoes": rec["safety"].get("vetoes", []),
                      "canary_loss_rel": round(rec["safety"].get("canary_loss_rel", float("nan")), 4),
                      "max_tensor_rel": round(rec["safety"].get("max_tensor_rel", float("nan")), 4)}}
    if not rec.get("vetoed"):
        c = rec["compare"]
        rep = json.loads((run_dir / "reports" / f"step{step}__iter_val_0.json").read_text())
        per = rep["per_item"]
        out["dev"] = {"abstain_rate": {k: _rate(per, (k,)) for k in C.ABS_KINDS + C.ANS_KINDS},
                      "abstain_gain_abs": round(c["target_abstain"]["delta"], 3),
                      "discrimination": {str(k): round(v, 3) for k, v in c["discrimination"].items()},
                      "clean_f1_delta": [round(c["ans1.token_f1"]["delta"], 4), [round(x, 4) for x in c["ans1.token_f1"]["ci"]]],
                      "false_abstain_delta": round(c["ans_all.false_abstain"]["delta"], 3),
                      "fabrication_delta": round(c["ans1.fabrication"]["delta"], 4),
                      "retention_pct": {k: round(100 * v["rel"], 3) for k, v in c.get("retention", {}).items()},
                      "guardrails": c["guardrails"], "guardrails_pass": c["guardrails_pass"]}
        items = _dev_ovm()
        _, tok, eos = checkpoints.load("qa")
        m = engine.materialize(R.base, delta)
        resp = batchgen.generate(m, tok, eos, [(i["messages"][0]["content"], i["messages"][1]["content"]) for i in items], C.GEN_BATCH)
        op = [{"id": it["id"], "kind": it["kind"], "seed": it["seed_raft_idx"], "resp": r,
               "scores": metrics.score_item(r, it["messages"][2]["content"], it["messages"][1]["content"], not it["ovm_absent"])}
              for it, r in zip(items, resp)]
        (run_dir / "reports" / f"step{step}__dev_ovm.json").write_text(json.dumps(op))
        D = _disc_ci(items, op)
        out["ovm"] = {"D_matched": [round(x, 3) for x in D], "abstain_rate": {k: _rate(op, (k,)) for k in ("ans1", "ans2", "abs1", "abs2", "del1", "del2")}}
        ex = []
        for kinds, want in ((("del1",), "absent, answered"), (("ans1",), "answerable, abstained")):
            for p, it in zip(op, items):
                if p["kind"] in kinds and (p["scores"]["abstain"] == (want.endswith("abstained"))) and len(ex) < (3 if want.startswith("absent") else 6):
                    u = it["messages"][1]["content"]
                    ex.append({"case": want, "question": u[u.rfind("Question"):][:200] if "Question" in u else u[-200:], "response": p["resp"][:160]})
        out["examples"] = ex
        del m; torch.cuda.empty_cache()
    with open(run_dir / "steps.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **out}, default=float) + "\n")
    return out


class _gpu_lock:
    """One evaluation on the GPU at a time (3 concurrent runs made each step ~15x slower). The OS releases
    the lock if the process dies."""
    def __enter__(self):
        import msvcrt
        OUT.mkdir(parents=True, exist_ok=True)
        self.f = open(OUT / "gpu.lock", "a+")
        t0 = time.time()
        while True:
            try:
                self.f.seek(0); msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1); break
            except OSError:
                time.sleep(5)
        if time.time() - t0 > 5:
            print(f"(waited {time.time() - t0:.0f}s for the GPU)", flush=True)
        return self

    def __exit__(self, *a):
        import msvcrt
        self.f.seek(0); msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1); self.f.close()


def _steps(run_dir):
    p = run_dir / "steps.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--brief", action="store_true")
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--action")
    ap.add_argument("--action-file")
    ap.add_argument("--final", type=int)
    a = ap.parse_args()
    run_dir = OUT / f"run_{a.run}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if a.brief:
        print(BRIEF); print(f"Budget used: {len(_steps(run_dir))}/{BUDGET}"); return
    if a.history:
        for s in _steps(run_dir):
            print(json.dumps({k: s[k] for k in s if k not in ("examples", "t")}, default=float))
        print(f"Budget used: {len(_steps(run_dir))}/{BUDGET}"); return
    if a.final is not None:
        st = {s["step"]: s for s in _steps(run_dir)}
        if a.final not in st:
            sys.exit(f"no step {a.final}")
        (run_dir / "final.json").write_text(json.dumps({"final_step": a.final, "action": st[a.final]["action"]}))
        print(f"final = step {a.final} recorded"); return
    if len(_steps(run_dir)) >= BUDGET:
        sys.exit(f"BUDGET EXHAUSTED ({BUDGET}). Nominate a final step with --final <step>.")
    action = json.loads(open(a.action_file, encoding="utf-8").read() if a.action_file else a.action)
    from . import checkpoints
    checkpoints.verify_lock()
    try:
        build_delta(action, engine.load_state("qa"))
    except (ValueError, KeyError, TypeError) as e:
        sys.exit(f"INVALID ACTION (not charged): {e}")
    with _gpu_lock():
        print(json.dumps(evaluate(action, run_dir), indent=1, default=float))


if __name__ == "__main__":
    main()

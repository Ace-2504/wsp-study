"""Final frozen evaluation (docs/19, pre-registered): every arm, once, on the frozen 2,000-item
instrument (data/eval, SHA-locked), with the Gate 1 harness unchanged (qa_eval: one item at a time,
greedy, 160 tokens, 0-shot). Experimenter only; nothing here feeds back into any edit.

    python -m wsp.final_eval run       # GPU, resumable (per-arm partial files)
    python -m wsp.final_eval report    # -> wsp_runs/final_eval/final_numbers.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import checkpoints, engine, metrics, paths, qa_eval
from .editor_step import _gpu_lock

OUT = paths.RUNS / "final_eval"
R = paths.RUNS
G5 = R / "gate5_editor"

ARMS = {  # name -> (delta path or None for θ_qa, kind)
    "theta_qa": (None, "baseline"),
    "andgate_theta1": (R / "gate3b_andgate" / "deltas" / "selected.safetensors", "pure WSP"),
    "lora_lr3e-4": (R / "gate2" / "deltas" / "lora_lr0.0003.safetensors", "LoRA fine-tune"),
    "fullft_lr1e-5": (R / "gate4_fullft" / "deltas" / "fullft_lr1e-05.safetensors", "full fine-tune"),
    "editor_A": (G5 / "run_A" / "final.safetensors", "editor-LLM (gate + FT slice)"),
    "editor_B": (G5 / "run_B" / "final.safetensors", "editor-LLM (gate + FT slice)"),
    "editor_C": (G5 / "run_C" / "final.safetensors", "editor-LLM (gate + FT slice)"),
    "random_R1": (G5 / "run_R1" / "final.safetensors", "random search (gate + FT slice)"),
}
# held-out pooled iter_val_1..3 abstention gains, for the transfer check (filled by report())
HELDOUT = {}


def _recursion_arm():
    """The last good recursive model θ_K (docs/18), only if K >= 2 and it passed every held-out guardrail."""
    g = R / "gate6_recursion" / "gate6_recursion_numbers.json"
    if not g.exists():
        return None
    lg = json.loads(g.read_text()).get("last_good")
    if lg and lg != "theta1":
        return f"recursion_{lg}", (R / "gate6_recursion" / "deltas" / f"{lg}.safetensors", "pure WSP, recursive")
    return None


def arms():
    a = dict(ARMS)
    r = _recursion_arm()
    if r:
        a[r[0]] = r[1]
    return a


def run():
    import torch
    checkpoints.verify_lock()
    rows, sha = qa_eval.load_eval()
    OUT.mkdir(parents=True, exist_ok=True)
    base = None
    for name, (dp, kind) in arms().items():
        if (OUT / f"{name}.json").exists():
            continue
        with _gpu_lock():
            torch.manual_seed(0)
            model, tok, eos = checkpoints.load("qa")
            if dp is not None:
                if base is None:
                    base = engine.load_state("qa")
                del model; torch.cuda.empty_cache()
                model = engine.materialize(base, engine.load_delta(dp))
            gen = qa_eval.Generator(model, tok, eos, 0)
            items = qa_eval.run(gen, rows, qa_eval.CONDS, OUT / f"{name}.partial.jsonl")
            del model, gen; torch.cuda.empty_cache()
        res = {"arm": name, "kind": kind, "delta": str(dp) if dp else None, "eval_sha256": sha,
               "conditions": metrics.aggregate(items)}
        (OUT / f"{name}.json").write_text(json.dumps({"result": res, "per_item": items}))
        for c, v in res["conditions"].items():
            print(f"[{name}] {c:18s} F1 {v['token_f1'][0]:.3f} fab {v['fabrication'][0]:.3f} abst {v['abstain_rate'][0]:.3f}", flush=True)


def _per(name, cond, key):
    d = json.loads((OUT / f"{name}.json").read_text())["per_item"]
    return {it["pair_id"]: it["scores"][key] for it in d if it["cond"] == cond}


def _heldout_gain(name):
    """Pooled iter_val_1..3 abstention gain measured earlier (for the transfer check)."""
    try:
        if name == "lora_lr3e-4":
            return json.loads((R / "gate2" / "gate2_numbers.json").read_text())["heldout"]["lora_lr0.0003"]["paired"]["target_abstain"]["delta"]
        if name == "fullft_lr1e-5":
            return json.loads((R / "gate4_fullft" / "gate4_fullft_numbers.json").read_text())["heldout"]["selected"]["target_abstain"]["delta"]
        if name.startswith("editor_") or name == "random_R1":
            f = "gate5_random_numbers.json" if name == "random_R1" else "gate5_editor_numbers.json"
            return json.loads((G5 / f).read_text())[name.split("_")[1]]["final"]["target_abstain"]["delta"]
        if name.startswith("recursion_") or name == "andgate_theta1":
            g = json.loads((R / "gate6_recursion" / "gate6_recursion_numbers.json").read_text())
            key = "theta1" if name == "andgate_theta1" else name.split("_")[1]
            return g["models"][key]["target_abstain"]["delta"]
    except (OSError, KeyError, TypeError):
        return None
    return None


def report():
    out = {}
    names = [n for n in arms() if (OUT / f"{n}.json").exists()]
    for n in names:
        res = json.loads((OUT / f"{n}.json").read_text())["result"]
        row = {"kind": res["kind"], "conditions": {c: {k: v[0] for k, v in m.items() if isinstance(v, list)}
                                                   for c, m in res["conditions"].items()}}
        if n != "theta_qa":
            row["rf_abstain"] = metrics.paired_delta_ci(_per(n, "retrieval_failure", "abstain"), _per("theta_qa", "retrieval_failure", "abstain"))
            row["clean_f1"] = metrics.paired_delta_ci(_per(n, "clean", "token_f1"), _per("theta_qa", "clean", "token_f1"))
            row["realistic_f1"] = metrics.paired_delta_ci(_per(n, "realistic", "token_f1"), _per("theta_qa", "realistic", "token_f1"))
            row["clean_fab"] = metrics.paired_delta_ci(_per(n, "clean", "fabrication"), _per("theta_qa", "clean", "fabrication"))
            fa = {**{("c", k): v for k, v in _per(n, "clean", "abstain").items()}, **{("r", k): v for k, v in _per(n, "realistic", "abstain").items()}}
            fq = {**{("c", k): v for k, v in _per("theta_qa", "clean", "abstain").items()}, **{("r", k): v for k, v in _per("theta_qa", "realistic", "abstain").items()}}
            row["false_abstain_clean_realistic"] = metrics.paired_delta_ci({str(k): v for k, v in fa.items()}, {str(k): v for k, v in fq.items()})
            row["guards"] = {"rf_abstain_up": row["rf_abstain"]["ci"][0] > 0,
                             "clean_f1": row["clean_f1"]["ci"][0] >= -0.02,
                             "fabrication": not (row["clean_fab"]["significant"] and row["clean_fab"]["delta"] > 0)}
            row["final_pass"] = all(row["guards"].values())
            h = _heldout_gain(n)
            row["heldout_abstain_gain"] = h
            if h:
                row["transfer_ratio"] = row["rf_abstain"]["delta"] / h
                row["transfers"] = row["rf_abstain"]["ci"][1] >= 0.5 * h
        out[n] = row
    (OUT / "final_numbers.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()

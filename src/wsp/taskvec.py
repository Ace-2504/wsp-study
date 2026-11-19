"""The QA task vector τ = θ_qa − θ_e4, characterised per weight matrix (CPU only, no model forward).

θ_e4 ships in fp32 and θ_qa in bf16, so τ mixes two things: what fine-tuning learned, and the
rounding of e4's fp32 weights to bf16. The rounding part ρ = bf16(θ_e4) − θ_e4 is computed
exactly, and the learned part is τ_learned = θ_qa − bf16(θ_e4) (both sides on the bf16 grid).
Per module we report ‖τ‖/‖θ_e4‖ and the share of ‖τ‖² that is pure rounding. Modules where
rounding dominates cannot support a meaningful per-module α in the task-vector arms.

    python -m wsp.taskvec
"""
from __future__ import annotations

import json

from . import checkpoints, paths


def module_group(name: str) -> str:
    """'model.layers.3.self_attn.q_proj.weight' -> 'L03.q_proj'; embeddings / norms kept by name."""
    p = name.split(".")
    if "layers" in p:
        i = p.index("layers")
        return f"L{int(p[i + 1]):02d}.{p[-2]}"
    return name.replace("model.", "").replace(".weight", "")


def main():
    import torch
    from safetensors.torch import load_file

    checkpoints.verify_lock()
    e4 = load_file(str(paths.ORIGINALS["e4"] / "model.safetensors"))
    qa = load_file(str(paths.ORIGINALS["qa"] / "model.safetensors"))
    assert e4.keys() == qa.keys(), set(e4) ^ set(qa)
    rows, tot = [], {"tau2": 0.0, "rho2": 0.0, "learn2": 0.0, "e42": 0.0}
    for k in sorted(e4):
        a = e4[k].double()
        b = qa[k].double()
        a16 = e4[k].to(torch.bfloat16).double()
        tau, rho, learn = b - a, a16 - a, b - a16
        r = {"param": k, "group": module_group(k), "numel": a.numel(),
             "norm_e4": a.norm().item(), "norm_tau": tau.norm().item(),
             "rel_tau": (tau.norm() / a.norm()).item(),
             "rounding_share_of_tau_sq": (rho.norm() ** 2 / tau.norm() ** 2).item() if tau.norm() > 0 else None,
             "frac_entries_changed_beyond_rounding": (learn != 0).double().mean().item()}
        rows.append(r)
        tot["tau2"] += tau.norm().item() ** 2; tot["rho2"] += rho.norm().item() ** 2
        tot["learn2"] += learn.norm().item() ** 2; tot["e42"] += a.norm().item() ** 2
    summary = {"n_tensors": len(rows), "n_params": sum(r["numel"] for r in rows),
               "global_rel_tau": (tot["tau2"] / tot["e42"]) ** .5,
               "global_rounding_share_of_tau_sq": tot["rho2"] / tot["tau2"],
               "global_rel_learned": (tot["learn2"] / tot["e42"]) ** .5,
               "n_coefficients_if_one_alpha_per_tensor": len(rows)}
    out = paths.RUNS / "gate1" / "taskvec_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_tensor": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    worst = sorted((r for r in rows if r["rounding_share_of_tau_sq"] is not None),
                   key=lambda r: -r["rounding_share_of_tau_sq"])[:6]
    print("highest rounding share:", [(r["param"], round(r["rounding_share_of_tau_sq"], 3)) for r in worst])
    big = sorted(rows, key=lambda r: -r["rel_tau"])[:6]
    print("largest relative change:", [(r["param"], round(r["rel_tau"], 4)) for r in big])


if __name__ == "__main__":
    main()

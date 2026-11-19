"""Parameter modification engine + safety gate + append-only run log.

Every arm is expressed as a DELTA: {parameter name -> ΔW}. Edits are only ever applied to an
in-memory copy of an original; the originals on disk are read-only and hash-locked.

    base  = load_state("qa")                 # CPU fp32 copy of θ_qa
    model = materialize(base, delta)         # new model = θ_qa + Δ  (never touches disk originals)
    gate  = SafetyGate(reference_model).check(delta, base, model)
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from . import checkpoints, config_wsp as C, paths

TIED = "lm_head.weight"   # tied to model.embed_tokens.weight; never edited separately


# ---------------- state / delta ops ----------------
def load_state(tag: str) -> dict:
    """fp32 CPU copy of an original's weights (tied lm_head excluded: it aliases the embedding)."""
    import torch
    from safetensors.torch import load_file
    sd = load_file(str(checkpoints.model_dir(tag) / "model.safetensors"))
    return {k: v.to(torch.float32) for k, v in sd.items() if k != TIED}


def norms(delta: dict, base: dict) -> dict:
    per = {k: float(d.norm() / base[k].norm()) for k, d in delta.items()}
    tot_d = sum(float(d.norm()) ** 2 for d in delta.values()) ** .5
    tot_b = sum(float(b.norm()) ** 2 for b in base.values()) ** .5
    return {"per_tensor_rel": per, "global_rel": tot_d / tot_b,
            "max_tensor_rel": max(per.values()) if per else 0.0,
            "n_tensors": len(delta), "n_params": int(sum(d.numel() for d in delta.values()))}


def materialize(base: dict, delta: dict, scale: float = 1.0, device: str = "cuda"):
    """Build a fresh bf16 model = base + scale·delta, IDENTICAL in construction to checkpoints.load():
    the model is instantiated directly in bf16 (so non-weight buffers such as RoPE inv_freq are
    computed exactly as in the reference), then the edited weights (summed in fp32, cast once to
    bf16) are loaded into it. A zero delta reproduces θ_qa bit-for-bit (gate2 check 1)."""
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(checkpoints.model_dir("qa"), torch_dtype=torch.bfloat16)
    sd = {k: ((v + scale * delta[k]) if k in delta else v).to(torch.bfloat16) for k, v in base.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert set(missing) <= {TIED} and not unexpected, (missing, unexpected)
    model.tie_weights()
    model = model.to(device).eval()
    model.config.eos_token_id = 1
    return model


def random_like(delta: dict, seed: int) -> dict:
    """Magnitude-matched random control: same tensors, Gaussian direction, identical per-tensor
    Frobenius norm. If ΔW is low-rank (LoRA), the control keeps the SAME rank, so rank and norm
    are both matched and only the direction differs."""
    import torch
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, d in delta.items():
        r = int(torch.linalg.matrix_rank(d.float()).item()) if d.dim() == 2 else None
        if d.dim() == 2 and r < min(d.shape):
            a = torch.randn(d.shape[0], r, generator=g); b = torch.randn(r, d.shape[1], generator=g)
            x = a @ b
        else:
            x = torch.randn(d.shape, generator=g)
        out[k] = x * (d.norm() / x.norm())
    return out


def save_delta(delta: dict, path: Path) -> str:
    from safetensors.torch import save_file
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in delta.items()}, str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_delta(path: Path) -> dict:
    from safetensors.torch import load_file
    return load_file(str(path))


# ---------------- safety gate ----------------
class SafetyGate:
    """Pre-evaluation checks with veto power. Reference = θ_qa (the edit's parent)."""

    def __init__(self, ref_model, canary_ids):
        self.canary_ids = canary_ids                      # LongTensor [n, 1024] of held-out text
        self.ref_act, self.ref_loss = self._probe(ref_model)

    def _probe(self, model):
        import torch
        import torch.nn.functional as F
        acts, losses = [], []
        with torch.no_grad():
            for s in range(0, len(self.canary_ids), 16):
                x = self.canary_ids[s:s + 16].cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = model(input_ids=x, output_hidden_states=True)
                acts.append(torch.stack([h.float().abs().amax() for h in o.hidden_states]))
                lg = o.logits.float()
                if not torch.isfinite(lg).all():
                    return None, float("inf")
                losses.append(F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1)).item())
        return torch.stack(acts).amax(0).cpu(), sum(losses) / len(losses)

    def check(self, delta: dict, base: dict, model) -> dict:
        import torch
        rep = {"pass": True, "vetoes": []}
        finite = all(torch.isfinite(d).all() for d in delta.values())
        rep["finite"] = bool(finite)
        if not finite:
            rep["pass"] = False; rep["vetoes"].append("non-finite delta"); return rep
        n = norms(delta, base)
        rep["global_rel"], rep["max_tensor_rel"] = n["global_rel"], n["max_tensor_rel"]
        if n["max_tensor_rel"] > C.MAX_REL_DELTA_PER_TENSOR:
            rep["pass"] = False; rep["vetoes"].append(f"tensor rel Δ {n['max_tensor_rel']:.3f} > {C.MAX_REL_DELTA_PER_TENSOR}")
        if n["global_rel"] > C.MAX_REL_DELTA_GLOBAL:
            rep["pass"] = False; rep["vetoes"].append(f"global rel Δ {n['global_rel']:.4f} > {C.MAX_REL_DELTA_GLOBAL}")
        act, loss = self._probe(model)
        if act is None:
            rep["pass"] = False; rep["vetoes"].append("non-finite logits on canary"); return rep
        ratio = float((act / self.ref_act.clamp_min(1e-6)).max())
        rep["max_activation_ratio"], rep["canary_loss_rel"] = ratio, loss / self.ref_loss - 1
        if ratio > C.MAX_ACTIVATION_RATIO:
            rep["pass"] = False; rep["vetoes"].append(f"activation ratio {ratio:.2f} > {C.MAX_ACTIVATION_RATIO}")
        if rep["canary_loss_rel"] > C.MAX_CANARY_LOSS_REL_INCREASE:
            rep["pass"] = False; rep["vetoes"].append(f"canary loss +{rep['canary_loss_rel']:.1%}")
        return rep


# ---------------- run log ----------------
class RunLog:
    """Append-only JSONL. One record per event; nothing is ever rewritten."""

    def __init__(self, exp: str):
        self.dir = paths.RUNS / exp
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "log.jsonl"

    def write(self, kind: str, **rec):
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": kind, **rec}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")
        return rec

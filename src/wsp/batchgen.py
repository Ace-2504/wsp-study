"""Batched greedy generation for the edit-eval loop.

Same prompt rendering as wsp.qa_eval (custom <|role|> template, add_special_tokens=False, no BOS),
but: items are sorted by prompt length and left-padded into batches WITH an explicit attention
mask, and max_new_tokens is capped at 1024 − prompt length so no position ever exceeds the
model's context (the Gate 1 harness inherited the reference's uncapped 160).

Batching changes bf16 arithmetic, so a small fraction of greedy outputs can differ from the
one-at-a-time harness. `agreement_check()` measures that rate; every arm in a comparison is
scored with this SAME procedure, so it is a fixed property of the instrument, not a confound.
"""
from __future__ import annotations

from .qa_eval import MAX_NEW, render

CTX = 1024


def generate(model, tok, eos: int, prompts: list[tuple[str, str]], batch_size: int = 32,
             device: str = "cuda") -> list[str]:
    """prompts: list of (system, user). Returns responses in input order."""
    import torch
    texts = [render(s, u, []) for s, u in prompts]
    enc = [tok(t, add_special_tokens=False).input_ids for t in texts]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    out = [None] * len(enc)
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        L = max(len(enc[i]) for i in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            n = len(enc[i]); ids[r, L - n:] = torch.tensor(enc[i]); att[r, L - n:] = 1
        new = max(1, min(MAX_NEW, CTX - L))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            g = model.generate(input_ids=ids.to(device), attention_mask=att.to(device),
                               max_new_tokens=new, do_sample=False, pad_token_id=pad,
                               eos_token_id=eos)
        for r, i in enumerate(idx):
            out[i] = tok.decode(g[r, L:], skip_special_tokens=True).strip()
    return out


def agreement_check(model, tok, eos, ref_items: list[dict], batch_size: int = 32) -> dict:
    """Compare batched outputs with stored one-at-a-time outputs (Gate 1 per_item)."""
    resp = generate(model, tok, eos, [(i["system"], i["user"]) for i in ref_items], batch_size)
    same = sum(a == b["resp"] for a, b in zip(resp, ref_items))
    return {"n": len(ref_items), "identical": same, "rate": same / max(1, len(ref_items)), "resp": resp}

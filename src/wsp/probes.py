"""Phase-0 copy / induction probes, ported from vendor/qa_probe/modal_phase0_probe.py to run locally.

Same seeds (1337), same consumption order of the RNGs, same templates, same passages (v1's
held-out val bins, sorted shard order), same scoring. Probe D (per-domain loss) is not repeated
here: wsp.retention computes it on the full sets and also reports the Phase-0 "D" protocol value.

These probes are pure continuation, no chat template. For the QA model they measure whether
fine-tuning preserved the copy machinery that grounded QA builds on, which is the "capability
control" from the proposal.

    python -m wsp.probes --model e4 --name e4
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from . import checkpoints, metrics, paths
from .retention import SEQ

SEED, N_INDUCTION, IND_LEN, N_NEEDLE, GEN_TOKENS = 1337, 64, 64, 200, 12
VOCAB = 16_384
TEMPLATES = [
    ("The internal reference number for this filing is {c}.",
     "The internal reference number for this filing is",
     "\nQuestion: What is the internal reference number for this filing?\nAnswer:"),
    ("The docket identifier assigned to this matter is {c}.",
     "The docket identifier assigned to this matter is",
     "\nQuestion: What is the docket identifier assigned to this matter?\nAnswer:"),
    ("The archive catalogue code for this record is {c}.",
     "The archive catalogue code for this record is",
     "\nQuestion: What is the archive catalogue code for this record?\nAnswer:"),
]


def _ci(vals):
    _, lo, hi = metrics.bootstrap_ci(vals)
    return [lo, hi]


def run(model, tok, device: str = "cuda") -> dict:
    import torch
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    pyrng = random.Random(SEED)
    out: dict = {}

    passages, per_source = [], {}
    for src in ("sec", "case-law", "fineweb-edu"):
        files = sorted(paths.RETENTION["v1_heldout"].glob(f"{src}-*.bin"))
        ds = np.concatenate([np.memmap(f, dtype=np.uint16, mode="r").reshape(-1, SEQ) for f in files])
        idx = rng.choice(len(ds), size=min(64, len(ds)), replace=False)
        ids = ds[np.sort(idx).astype(np.int64)].astype(np.int64)
        per_source[src] = ids
        for row in ids[:24]:
            passages.append(tok.decode(row.tolist(), skip_special_tokens=True))

    def induction(seqs):
        first, second = [], []
        with torch.no_grad():
            for i in range(0, len(seqs), 16):
                b = seqs[i:i + 16].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=b).logits
                corr = (logits[:, :-1].argmax(-1) == b[:, 1:]).float().cpu().numpy()
                first.append(corr[:, :IND_LEN - 1]); second.append(corr[:, IND_LEN - 1:2 * IND_LEN - 1])
        f, s = np.concatenate(first).mean(1), np.concatenate(second).mean(1)
        return {"acc_first_copy": float(f.mean()), "acc_second_copy": float(s.mean()),
                "second_copy_ci95": _ci(s), "gain": float(s.mean() - f.mean()), "n": int(len(s))}

    rand = torch.from_numpy(rng.integers(3, VOCAB, size=(N_INDUCTION, IND_LEN)).astype(np.int64))
    out["A_induction_random"] = induction(torch.cat([rand, rand], 1))
    nat = [row[:IND_LEN] for ids in per_source.values() for row in ids[:24]]
    nat = torch.from_numpy(np.stack(nat[:N_INDUCTION]).astype(np.int64))
    out["B_induction_natural"] = induction(torch.cat([nat, nat], 1))

    def code():
        return f"{pyrng.choice('XKQRVZ')}{pyrng.choice('XKQRVZ')}-{pyrng.randint(1000, 9999)}"

    def needle(with_needle: bool, mode: str):
        hits, items = [], []
        for i in range(N_NEEDLE):
            words = passages[i % len(passages)].split()
            if len(words) < 120:
                continue
            c = code()
            sent, cue_v, cue_q = TEMPLATES[i % len(TEMPLATES)]
            mid = len(words) // 2
            body = words[:mid] + ([sent.format(c=c)] if with_needle else []) + words[mid:]
            prompt = " ".join(body) + ("\n" + cue_v if mode == "verbatim" else cue_q)
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
            ids = ids[:, -(SEQ - GEN_TOKENS):].to(device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                g = model.generate(ids, max_new_tokens=GEN_TOKENS, do_sample=False, pad_token_id=2)
            cont = tok.decode(g[0, ids.shape[1]:].tolist(), skip_special_tokens=True)
            hits.append(1.0 if c in cont.replace(" ", "") or c in cont else 0.0)
            items.append({"i": i, "code": c, "hit": hits[-1], "got": cont[:40]})
        return {"rate": float(np.mean(hits)) if hits else float("nan"), "ci95": _ci(hits),
                "n": len(hits), "per_item": items}

    out["C_needle_verbatim"] = needle(True, "verbatim")
    out["C_control_no_needle"] = needle(False, "verbatim")
    out["C2_needle_question"] = needle(True, "question")
    out["C2_control_no_needle"] = needle(False, "question")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default=str(paths.RUNS / "gate1" / "probes"))
    a = ap.parse_args()
    checkpoints.verify_lock()
    model, tok, _ = checkpoints.load(a.model)
    res = run(model, tok)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.name}.json").write_text(json.dumps({"model": a.model, **res}, indent=1))
    for k, v in res.items():
        brief = {kk: vv for kk, vv in v.items() if kk != "per_item"}
        print(f"[{a.name}] {k}: {brief}", flush=True)


if __name__ == "__main__":
    main()

"""Phase 0 — capability gate for v1-extended QA-SFT.

Answers ONE question before any money is spent on QA generation:
**can this model copy information out of its context at all?**

If it cannot, grounded SFT is known-futile in advance and the ceiling is architectural.
The previous QA-SFT study (v1->v2, 5 rounds) never ran this check and consequently could
not distinguish "task design failed" from "model was never capable".

v1-extended is a BASE model — it continues text, it does not follow instructions. So every
probe here is pure continuation. Nothing asks the model to obey a request.

  A  induction (random tokens)  — pure copy machinery, no linguistic priors
  B  induction (natural text)   — the same, in-distribution
  C  needle retrieval           — targeted copy of a specific span == the grounding skill
  D  per-domain baseline loss   — the zero point for every later forgetting number

Probe C is the plan's headline "copy rate". C_control (needle absent) is the guessing floor.

    modal run modal_phase0_probe.py
"""

from __future__ import annotations

import modal

app = modal.App("slm-125m-phase0-probe")
volume = modal.Volume.from_name("slm-125m")

V1_CKPT = "/data/checkpoints/base/ckpt.pt"
V1EXT_CKPT = "/data/cont/checkpoints/base/ckpt.pt"
TOKENIZER_DIR = "/data/tokenizer"
V1_VAL_ROOT = "/data/v1_val"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "numpy>=1.26,<2.0",
    )
    .add_local_python_source("config", "train_core")
)

SEED = 1337
N_INDUCTION = 64        # sequences per induction probe
IND_LEN = 64            # tokens per copy
N_NEEDLE = 200          # needle trials (drives the CI on the headline number)
GEN_TOKENS = 12


def _bootstrap_ci(vals, n=5000, seed=0):
    """95% CI on a mean. Every number in this study ships with an interval — the previous
    study's fatal flaw was point estimates on a metric too noisy to support them."""
    import numpy as np
    rng = np.random.default_rng(seed)
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return (float("nan"), float("nan"))
    means = rng.choice(a, size=(n, a.size), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


@app.function(image=image, volumes={"/data": volume}, gpu="L4", timeout=60 * 40)
def probe(ckpt_path: str = V1EXT_CKPT) -> dict:
    import json
    import random
    import re

    import numpy as np
    import torch
    from transformers import AutoTokenizer

    import config
    import train_core

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    pyrng = random.Random(SEED)
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    model = train_core._build_model(device, attn_implementation="sdpa")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    out: dict = {
        "ckpt": ckpt_path,
        "ckpt_step": int(ckpt.get("step", 0)),
        "ckpt_tokens_seen": int(ckpt.get("tokens_seen", 0)),
    }
    print(f"loaded {ckpt_path}: step {out['ckpt_step']}, "
          f"{out['ckpt_tokens_seen']:,} tokens seen", flush=True)

    # ---- real held-out text (v1's val bins; never trained on by either model) ---------
    from train_core import BinWindowDataset
    passages: list[str] = []
    per_source_ids: dict[str, np.ndarray] = {}
    for src in ("sec", "case-law", "fineweb-edu"):
        try:
            ds = BinWindowDataset(V1_VAL_ROOT, "val", config.SEQ_LEN, source_prefix=src)
        except FileNotFoundError:
            print(f"  [warn] no val bins for {src}", flush=True)
            continue
        idx = rng.choice(len(ds), size=min(64, len(ds)), replace=False)
        ids = ds.get_batch(np.sort(idx).astype(np.int64), "cpu").numpy()
        per_source_ids[src] = ids
        for row in ids[:24]:
            passages.append(tok.decode(row.tolist(), skip_special_tokens=True))
    print(f"decoded {len(passages)} held-out passages from {list(per_source_ids)}", flush=True)

    # =================================================================================
    # A + B — induction: feed [X, X] and measure next-token accuracy on the 2nd copy.
    # A copy-capable model predicts the repeat almost perfectly; a model without
    # induction heads shows no gain over the first copy.
    # =================================================================================
    def induction(seqs: torch.Tensor) -> dict:
        first, second = [], []
        with torch.no_grad():
            for i in range(0, len(seqs), 16):
                batch = seqs[i:i + 16].to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits = model(input_ids=batch).logits
                pred = logits[:, :-1].argmax(-1)
                corr = (pred == batch[:, 1:]).float().cpu().numpy()
                first.append(corr[:, :IND_LEN - 1])
                second.append(corr[:, IND_LEN - 1:2 * IND_LEN - 1])
        f = np.concatenate(first).mean(axis=1)
        s = np.concatenate(second).mean(axis=1)
        lo, hi = _bootstrap_ci(s)
        return {"acc_first_copy": float(f.mean()), "acc_second_copy": float(s.mean()),
                "second_copy_ci95": [lo, hi], "gain": float(s.mean() - f.mean()),
                "n": int(len(s))}

    rand_tok = torch.from_numpy(
        rng.integers(3, config.MODEL.vocab_size, size=(N_INDUCTION, IND_LEN)).astype(np.int64))
    out["A_induction_random"] = induction(torch.cat([rand_tok, rand_tok], dim=1))
    print(f"A random-token induction: {out['A_induction_random']}", flush=True)

    nat = []
    for src_ids in per_source_ids.values():
        for row in src_ids[:24]:
            nat.append(row[:IND_LEN])
    nat_tok = torch.from_numpy(np.stack(nat[:N_INDUCTION]).astype(np.int64))
    out["B_induction_natural"] = induction(torch.cat([nat_tok, nat_tok], dim=1))
    print(f"B natural-text induction: {out['B_induction_natural']}", flush=True)

    # =================================================================================
    # C — needle retrieval. Plant an unguessable code in a real passage, then end the
    # prompt on that sentence's prefix. Reproducing the code = using the context.
    # C_control removes the needle: anything scored there is guessing, not retrieval.
    # =================================================================================
    def make_code() -> str:
        return (f"{pyrng.choice('XKQRVZ')}{pyrng.choice('XKQRVZ')}"
                f"-{pyrng.randint(1000, 9999)}")

    # Each template gives the planted sentence, a VERBATIM cue (literal string repeat ->
    # tests induction-head copying) and a QUESTION cue (tests question->span mapping, the
    # skill SFT would actually teach). The gap between them is the whole point: if copying
    # is high and question-form is low, the bottleneck is the mapping, not the machinery.
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

    def needle_trials(with_needle: bool, cue_mode: str = "verbatim") -> dict:
        hits, trials, prompts_used = [], [], 0
        for i in range(N_NEEDLE):
            text = passages[i % len(passages)]
            words = text.split()
            if len(words) < 120:
                continue
            code = make_code()
            sent, cue_v, cue_q = TEMPLATES[i % len(TEMPLATES)]
            cue = cue_v if cue_mode == "verbatim" else cue_q
            mid = len(words) // 2
            body = words[:mid] + ([sent.format(c=code)] if with_needle else []) + words[mid:]
            prompt = " ".join(body) + ("\n" + cue if cue_mode == "verbatim" else cue)
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
            ids = ids[:, -(config.SEQ_LEN - GEN_TOKENS):].to(device)
            with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                gen = model.generate(ids, max_new_tokens=GEN_TOKENS, do_sample=False,
                                     pad_token_id=2)
            cont = tok.decode(gen[0, ids.shape[1]:].tolist(), skip_special_tokens=True)
            hits.append(1.0 if code in cont.replace(" ", "") or code in cont else 0.0)
            trials.append({"i": i, "code": code, "hit": hits[-1]})
            prompts_used += 1
            if i < 3 and with_needle:
                print(f"  needle[{cue_mode}][{i}] want={code!r} got={cont[:60]!r}", flush=True)
        lo, hi = _bootstrap_ci(hits)
        # Per-item results are PERSISTED. Both models see identical seeds -> identical
        # passages and codes -> trials are paired, so a paired CI on the difference is
        # computable later. (The previous study discarded per-item scores and could never
        # error-bar anything retrospectively; not repeating that.)
        return {"exact_match_rate": float(np.mean(hits)) if hits else float("nan"),
                "ci95": [lo, hi], "n": prompts_used, "per_item": trials}

    def _brief(d):
        return {k: v for k, v in d.items() if k != "per_item"}

    out["C_needle_retrieval"] = needle_trials(True, "verbatim")
    print(f"C verbatim-cue retrieval: {_brief(out['C_needle_retrieval'])}", flush=True)
    out["C_control_no_needle"] = needle_trials(False, "verbatim")
    print(f"C control (guessing floor): {_brief(out['C_control_no_needle'])}", flush=True)
    out["C2_question_cue"] = needle_trials(True, "question")
    print(f"C2 question-cue retrieval: {_brief(out['C2_question_cue'])}", flush=True)
    out["C2_control_no_needle"] = needle_trials(False, "question")
    print(f"C2 control (guessing floor): {_brief(out['C2_control_no_needle'])}", flush=True)

    # =================================================================================
    # D — per-domain baseline loss. The zero point every later forgetting claim needs.
    # =================================================================================
    dom = {}
    for src in ("sec", "case-law", "fineweb-edu"):
        try:
            ds = BinWindowDataset(V1_VAL_ROOT, "val", config.SEQ_LEN, source_prefix=src)
        except FileNotFoundError:
            continue
        loss = train_core.evaluate_loss(model, ds, batch_size=8, device=device, max_batches=40)
        dom[src] = {"loss": float(loss), "ppl": float(np.exp(loss))}
        print(f"D {src}: loss {loss:.4f} ppl {np.exp(loss):.2f}", flush=True)
    out["D_domain_baseline"] = dom

    # ---- gate verdict (thresholds fixed in SFT_PLAN.md §3, before this ran) ----------
    cr = out["C_needle_retrieval"]["exact_match_rate"]
    out["gate"] = {
        "copy_rate": cr,
        "verdict": ("PROCEED" if cr >= 0.60
                    else "PROCEED_CAPACITY_CAPPED" if cr >= 0.20
                    else "STOP_CAPACITY_CEILING"),
    }
    print(f"\nGATE: copy_rate={cr:.3f} -> {out['gate']['verdict']}", flush=True)
    return out


@app.local_entrypoint()
def main(ckpt_path: str = V1EXT_CKPT, compare_v1: bool = False):
    import json

    results = {"v1-extended": probe.remote(ckpt_path)}
    if compare_v1:
        results["v1"] = probe.remote(V1_CKPT)
    with open("phase0_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nwrote phase0_results.json")

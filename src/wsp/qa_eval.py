"""Deterministic grounded-QA harness over the frozen 2,000-item instrument.

Faithful to vendor/finetuning/eval.py (custom family):
  * prompt  = <|system|>\n{sys}<|eos|>\n [ <|user|>\n{u}<|eos|>\n<|assistant|>\n{a}<|eos|>\n ]*k
              <|user|>\n{user}<|eos|>\n<|assistant|>\n
  * tokenised with add_special_tokens=False (the tokenizer's add_bos_token=True would otherwise
    prepend the never-trained <|bos|>)
  * k-shot exemplars = the first k (user, assistant) pairs of qa_sft.jsonl, dropped one at a time
    until the prompt fits 1024 - 160 - 8 tokens; k=3 for base models, 0 for tuned ones
  * ONE item at a time, bf16 weights + bf16 autocast, greedy, 160 new tokens, eos = <|eos|>
    (do_sample=False is passed explicitly: the shipped generation_config.json says do_sample=True)

Resumable: every generation is appended + fsync'd to <out>.partial.jsonl; a re-run skips finished
(cond, pair_id) keys. The final JSON uses the vendor's {result, per_item} schema.

    python -m wsp.qa_eval --model qa --name qa_clean --conds clean
    python -m wsp.qa_eval --model e4 --name e4_all --few-shot 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

from . import checkpoints, metrics, paths

CONDS = ("clean", "realistic", "retrieval_failure", "closed_book")
MAX_NEW = 160


def load_eval() -> tuple[list[dict], str]:
    raw = paths.EVAL_MESSAGES.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != paths.EVAL_SHA256:
        raise RuntimeError(f"eval instrument changed: {sha[:12]} != {paths.EVAL_SHA256[:12]}")
    return [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()], sha


def few_shot_pool(k: int) -> list[tuple[str, str]]:
    shots = []
    if k <= 0:
        return shots
    for line in open(paths.QA_SFT, encoding="utf-8"):
        if not line.strip():
            continue
        m = json.loads(line).get("messages", [])
        if len(m) >= 3 and m[1]["role"] == "user" and m[2]["role"] == "assistant":
            shots.append((m[1]["content"], m[2]["content"]))
        if len(shots) >= k:
            break
    return shots


def render(system: str, user: str, shots) -> str:
    parts = [f"<|system|>\n{system}<|eos|>\n"]
    for u, a in shots:
        parts.append(f"<|user|>\n{u}<|eos|>\n<|assistant|>\n{a}<|eos|>\n")
    parts.append(f"<|user|>\n{user}<|eos|>\n<|assistant|>\n")
    return "".join(parts)


class Generator:
    """Wraps a loaded model so the WSP loop can reuse it with in-memory (edited) weights."""

    def __init__(self, model, tok, eos: int, k_shot: int, device: str = "cuda"):
        self.model, self.tok, self.eos, self.device = model, tok, eos, device
        self.shots = few_shot_pool(k_shot)
        self.k_shot = k_shot
        ctx = getattr(model.config, "max_position_embeddings", 1024) or 1024
        self.budget = ctx - MAX_NEW - 8
        self.pad = tok.pad_token_id or eos

    def __call__(self, system: str, user: str) -> tuple[str, dict]:
        import torch
        k = len(self.shots)
        while True:
            ids = self.tok(render(system, user, self.shots[:k]), return_tensors="pt",
                           add_special_tokens=False).input_ids.to(self.device)
            if ids.shape[1] <= self.budget or k == 0:
                break
            k -= 1
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                                      pad_token_id=self.pad, eos_token_id=self.eos)
        resp = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        return resp, {"prompt_tokens": int(ids.shape[1]), "shots_used": k,
                      "over_ctx": bool(ids.shape[1] + MAX_NEW > 1024)}


def run(gen: Generator, rows: list[dict], conds, partial: Path, log_every: int = 50,
        limit: int = 0) -> list[dict]:
    done = {}
    if partial.exists():
        for line in open(partial, encoding="utf-8"):
            if line.strip():
                it = json.loads(line)
                done[(it["cond"], it["pair_id"])] = it
    todo = [r for r in rows if r["meta"]["cond"] in conds]
    if limit:
        by = {}
        for r in todo:
            by.setdefault(r["meta"]["cond"], [])
            if len(by[r["meta"]["cond"]]) < limit:
                by[r["meta"]["cond"]].append(r)
        todo = [r for c in conds for r in by.get(c, [])]
    t0, n_new = time.time(), 0
    with open(partial, "a", encoding="utf-8") as fh:
        for r in todo:
            key = (r["meta"]["cond"], r["meta"]["pair_id"])
            if key in done:
                continue
            system, user, ref = (r["messages"][i]["content"] for i in range(3))
            answerable = metrics.ABSTAIN not in ref.lower()
            resp, info = gen(system, user)
            it = {"cond": key[0], "pair_id": key[1], "source": r["meta"].get("source"),
                  "answerable": answerable, "system": system, "user": user, "ref": ref,
                  "resp": resp, "scores": metrics.score_item(resp, ref, user, answerable), **info}
            fh.write(json.dumps(it) + "\n"); fh.flush(); os.fsync(fh.fileno())
            done[key] = it; n_new += 1
            if n_new % log_every == 0:
                rate = n_new / (time.time() - t0)
                print(f"  {n_new}/{len(todo)} new items  {rate:.2f} it/s", flush=True)
    return [done[(r["meta"]["cond"], r["meta"]["pair_id"])] for r in todo
            if (r["meta"]["cond"], r["meta"]["pair_id"]) in done]


def main():
    import torch
    import transformers

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="e4 | qa | path to a checkpoint dir")
    ap.add_argument("--name", required=True)
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--few-shot", type=int, default=-1, help="-1 auto: 3 for e4, else 0")
    ap.add_argument("--limit", type=int, default=0, help="per-condition cap (smoke tests)")
    ap.add_argument("--out-dir", default=str(paths.RUNS / "gate1"))
    a = ap.parse_args()

    torch.manual_seed(0)
    lock = checkpoints.verify_lock()
    rows, eval_sha = load_eval()
    conds = [c for c in a.conds.split(",") if c]
    k = (3 if a.model == "e4" else 0) if a.few_shot < 0 else a.few_shot
    model, tok, eos = checkpoints.load(a.model)
    gen = Generator(model, tok, eos, k)
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / f"{a.name}.partial.jsonl"
    t0 = time.time()
    items = run(gen, rows, conds, partial, limit=a.limit)
    result = {"version": a.name, "family": "custom", "conditions": metrics.aggregate(items),
              "manifest": {
                  "eval_sha256": eval_sha, "n_items": len(items), "conds": conds,
                  "decoding": {"preset": "plain", "max_new_tokens": MAX_NEW, "greedy": True},
                  "few_shot_k": k, "model": a.model,
                  "model_source": lock.get(a.model, {"path": a.model}),
                  "libs": {"torch": torch.__version__, "transformers": transformers.__version__},
                  "device": torch.cuda.get_device_name(0), "host": platform.node(),
                  "wall_s": round(time.time() - t0, 1),
                  "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
    out = out_dir / f"{a.name}.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"result": result, "per_item": items}, indent=1), encoding="utf-8")
    os.replace(tmp, out)
    for c, v in result["conditions"].items():
        print(f"[{a.name}] {c:18s} n={v['n']:3d} F1 {v['token_f1'][0]:.3f} EM {v['exact_match'][0]:.3f} "
              f"fab {v['fabrication'][0]:.3f} abst {v['abstain_rate'][0]:.3f}", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

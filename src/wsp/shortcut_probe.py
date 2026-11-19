"""Shortcut test: does an edit learn ABSTENTION, or does it learn "2 documents -> abstain"?

In both the item pools and the frozen eval, document count perfectly identifies the condition
(clean = 1 doc answerable, retrieval_failure = 2 docs absent, realistic = 3 docs answerable).
This probe fills the other cells of the 2 x 3 grid from held-out seeds (iter_val_1..3 only):

                1 doc            2 docs                 3 docs
  answerable    clean_like       ANS2 golden+D1 (new)   realistic_like
  absent        ABS1 D1 (new)    rf_like                ABS3 D1+D2+D3 (new)

D3 = a distractor taken from another seed of the SAME split, rejected if it contains the answer.
An edit that learned abstention abstains on the whole bottom row and answers the top row.

    python -m wsp.shortcut_probe --arms qa lora_lr0.0001 lora_lr0.0003 random_s0
"""
from __future__ import annotations

import argparse
import json
import random
import re

from . import batchgen, checkpoints, engine, metrics, paths, splits

OUT = paths.RUNS / "gate2"


def _docs(user):
    return [d.strip("\n") for d in re.split(r"\[Document \d+\]\n", user.rsplit("\n\nQuestion:", 1)[0]) if d.strip()]


def build() -> list[dict]:
    rng = random.Random(1337)
    items = []
    for f in (1, 2, 3):
        rows = splits.load(f"iter_val_{f}")
        seeds = {}
        for r in rows:
            seeds.setdefault(r["seed_raft_idx"], {})[r["kind"]] = r
        keys = sorted(seeds)
        for n, k in enumerate(keys):
            s = seeds[k]
            sys_, q = s["clean_like"]["messages"][0]["content"], splits.question_of(s["clean_like"]["messages"][1]["content"])
            ans = s["clean_like"]["messages"][2]["content"]
            gold = _docs(s["clean_like"]["messages"][1]["content"])[0]
            d1, d2 = _docs(s["rf_like"]["messages"][1]["content"])
            d3 = None
            for j in range(1, len(keys)):
                cand = _docs(seeds[keys[(n + j) % len(keys)]]["rf_like"]["messages"][1]["content"])[0]
                if cand not in (d1, d2, gold) and not (len(splits.norm(ans)) >= 4 and splits.norm(ans) in splits.norm(cand)):
                    d3 = cand; break
            two = [gold, d1]; rng.shuffle(two)
            for kind, docs, tgt in (("ans2", two, ans), ("abs1", [d1], splits.ABSTAIN), ("abs3", [d1, d2, d3], splits.ABSTAIN)):
                if docs[-1] is None:
                    continue
                items.append({"id": f"{k}:{kind}", "kind": kind, "system": sys_,
                              "user": splits.render_user(docs, q), "ref": tgt})
            for kind, src in (("ans1", "clean_like"), ("abs2", "rf_like"), ("ans3", "realistic_like")):
                m = s[src]["messages"]
                items.append({"id": f"{k}:{kind}", "kind": kind, "system": m[0]["content"], "user": m[1]["content"], "ref": m[2]["content"]})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    a = ap.parse_args()
    checkpoints.verify_lock()
    items = build()
    base = engine.load_state("qa")
    _, tok, eos = checkpoints.load("qa")
    res = {}
    for arm in a.arms:
        model = (checkpoints.load("qa")[0] if arm == "qa"
                 else engine.materialize(base, engine.load_delta(OUT / "deltas" / f"{arm}.safetensors")))
        resp = batchgen.generate(model, tok, eos, [(i["system"], i["user"]) for i in items])
        per = [{"id": i["id"], "kind": i["kind"], "abstain": metrics.is_abstain(r),
                "token_f1": metrics.token_f1(r, i["ref"])} for i, r in zip(items, resp)]
        cells = {}
        for k in ("ans1", "ans2", "ans3", "abs1", "abs2", "abs3"):
            rows = [p for p in per if p["kind"] == k]
            cells[k] = {"n": len(rows), "abstain": metrics.bootstrap_ci(p["abstain"] for p in rows),
                        "token_f1": metrics.bootstrap_ci(p["token_f1"] for p in rows)}
        res[arm] = {"cells": cells, "per_item": per}
        print(f"{arm:16s} abstain rate | answerable 1/2/3 docs: " +
              " ".join(f"{cells[k]['abstain'][0]:.3f}" for k in ("ans1", "ans2", "ans3")) +
              " | absent 1/2/3 docs: " + " ".join(f"{cells[k]['abstain'][0]:.3f}" for k in ("abs1", "abs2", "abs3")), flush=True)
        del model
    (OUT / "shortcut_probe.json").write_text(json.dumps(res))
    engine.RunLog("gate2").write("shortcut_probe", arms=a.arms, n_items=len(items),
                                 cells={k: v["cells"] for k, v in res.items()})


if __name__ == "__main__":
    main()

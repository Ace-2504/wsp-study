"""Train/eval 13-gram overlap: the edit-dev quarantine list and per-eval-item contamination scores.

Writes (small, tracked) files under splits/:
    qa_sft_quarantine.json  - qa_sft row indices sharing ANY 13-gram with an eval clean passage
                              (excluded from edit-dev, so no editing decision ever sees eval text)
    edit_dev.json           - the remaining qa_sft row indices = the WSP edit-dev pool
    eval_overlap.json       - per eval pair_id: fraction of its clean passage's 13-grams that occur
                              anywhere in qa_sft (used to test whether published scores are inflated)

    python -m wsp.overlap
"""
from __future__ import annotations

import hashlib
import json
import re

from . import paths

N = 13


def grams(text: str, n: int = N) -> set[str]:
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def passage(user: str) -> str:
    return user.rsplit("\n\nQuestion:", 1)[0]


def main():
    ev = [json.loads(l) for l in open(paths.EVAL_MESSAGES, encoding="utf-8")]
    tr = [json.loads(l) for l in open(paths.QA_SFT, encoding="utf-8")]
    ev_g = {r["meta"]["pair_id"]: grams(passage(r["messages"][1]["content"]))
            for r in ev if r["meta"]["cond"] == "clean"}
    ev_union = set().union(*ev_g.values())

    tr_g, quarantine = [], []
    for i, r in enumerate(tr):
        g = grams(" ".join(m["content"] for m in r["messages"] if m["role"] != "system"))
        tr_g.append(g)
        sh = g & ev_union
        if sh:
            quarantine.append({"idx": i, "chunk_id": r["meta"].get("chunk_id"),
                               "frac_own_grams_shared": round(len(sh) / max(1, len(g)), 4)})
    tr_union = set().union(*tr_g)
    eval_overlap = {pid: round(len(g & tr_union) / max(1, len(g)), 4) for pid, g in ev_g.items()}

    q_idx = {q["idx"] for q in quarantine}
    edit_dev = [i for i in range(len(tr)) if i not in q_idx]
    paths.SPLITS.mkdir(parents=True, exist_ok=True)
    src_sha = hashlib.sha256(paths.QA_SFT.read_bytes()).hexdigest()
    meta = {"source": "data/train/qa_sft.jsonl", "source_sha256": src_sha, "ngram": N,
            "eval_sha256": paths.EVAL_SHA256}
    (paths.SPLITS / "qa_sft_quarantine.json").write_text(json.dumps(
        {**meta, "n": len(quarantine), "rows": quarantine}, indent=1))
    (paths.SPLITS / "edit_dev.json").write_text(json.dumps(
        {**meta, "n": len(edit_dev), "rule": "qa_sft rows sharing no 13-gram with any eval clean passage",
         "idx": edit_dev}))
    (paths.SPLITS / "eval_overlap.json").write_text(json.dumps(
        {**meta, "metric": "fraction of the eval clean passage's 13-grams found anywhere in qa_sft",
         "by_pair_id": eval_overlap}, indent=1))

    b = {"0": 0, "(0,0.1)": 0, "[0.1,0.5)": 0, ">=0.5": 0}
    for f in eval_overlap.values():
        b["0" if f == 0 else "(0,0.1)" if f < .1 else "[0.1,0.5)" if f < .5 else ">=0.5"] += 1
    print(f"quarantined qa_sft rows: {len(quarantine)}  edit_dev: {len(edit_dev)}")
    print(f"eval clean items by overlap with qa_sft: {b}")


if __name__ == "__main__":
    main()

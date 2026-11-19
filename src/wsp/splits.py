"""Abstention + answerable item pools rebuilt from raft.jsonl, with NO eval document anywhere.

Why rebuilt: every eval retrieval_failure document also occurs inside RAFT prompts (as a
distractor), so RAFT rows cannot be used as-is. The answer-bearing passages and the questions do
not leak (1/500 clean passages, 0 question strings), so each RAFT answerable row keeps its
teacher-written question and golden passage, and gets NEW distractors from a clean document pool.

Rules (pre-registered at Gate 2, before any arm runs):
  1. Document pool = every distinct document text inside RAFT prompts. A document is TAINTED if
     >=50% of its 13-grams occur in any eval document (any condition); tainted documents are
     never used. (A single shared 13-gram is usually boilerplate, so it is not the criterion.)
  2. Seeds = RAFT answerable rows whose golden passage is locatable (contains the quoted span)
     and untainted. Rows are grouped by chunk_id and whole groups are assigned by hash:
     60% edit_dev, 40% iter_val in 4 rotation folds (fold 0 = Gate 2 development fold).
  3. Each document inherits a HOME split if it is some seed's golden passage. Distractors for an
     item may only come from documents whose home is the item's own split, or that are no seed's
     golden, so a passage that answers an iter_val question never appears in an edit_dev item.
  4. Distractors = the top-scoring clean, allowed documents by IDF-weighted overlap with the
     question (hard negatives, like the eval's similarity-picked distractors), excluding any
     document that contains the quoted span or the normalised short answer.
  5. DOCUMENT COUNT IS DECORRELATED FROM ANSWERABILITY (v2, after Gate 2a showed a LoRA learning
     "2 documents -> abstain": in the eval and in raft.jsonl the absent condition is always 2 docs).
     Each seed yields six items in the eval's exact format, three distractors D1..D3 per seed:
       ans1 [golden]                ans2 golden+D1 (shuffled)    ans3 golden+D1+D2 (shuffled) -> short answer
       abs1 [D1]                    abs2 D1+D2                   abs3 D1+D2+D3                -> "not stated in the context"
     so every document count is equally often answerable and absent.
     short answer = the text after RAFT's ##end_quote## (matches θ_qa's style and the eval gold).
  6. The final-test instrument (data/eval) is never touched.

Outputs: data/derived/item_pool.jsonl (regenerable, gitignored) and the tracked manifest
splits/item_pool_manifest.json (rules, counts, hashes).

    python -m wsp.splits
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import random
import re

from . import paths
from .overlap import grams

ABSTAIN = "not stated in the context"
EDIT_DEV_FRAC, N_FOLDS, TAINT = 0.60, 4, 0.50
POOL = paths.DATA / "derived" / "item_pool.jsonl"
MANIFEST = paths.SPLITS / "item_pool_manifest.json"
STOP = set("the a an of to in and or for on at by is are was were be with as that this it what which who "
           "how why when does did do from its their his her they he she has have had not under".split())


def docs_of(user: str) -> list[str]:
    body = user.rsplit("\n\nQuestion:", 1)[0]
    return [d.strip("\n") for d in re.split(r"\[Document \d+\]\n", body) if d.strip()]


def question_of(user: str) -> str:
    return user.rsplit("\n\nQuestion:", 1)[1].strip()


def render_user(docs: list[str], q: str) -> str:
    return "\n\n".join(f"[Document {i + 1}]\n{d}" for i, d in enumerate(docs)) + f"\n\nQuestion: {q}"


def norm(s: str) -> str:
    return " ".join(re.findall(r"\w+", s.lower()))


def words(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2]


def split_of(chunk_id: str) -> str:
    b = int(hashlib.sha256(f"wsp-gate2:{chunk_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if b < EDIT_DEV_FRAC:
        return "edit_dev"
    return f"iter_val_{min(N_FOLDS - 1, int((b - EDIT_DEV_FRAC) / ((1 - EDIT_DEV_FRAC) / N_FOLDS)))}"


def main():
    ev = [json.loads(l) for l in open(paths.EVAL_MESSAGES, encoding="utf-8")]
    rf = [json.loads(l) for l in open(paths.RAFT, encoding="utf-8")]
    system = rf[0]["messages"][0]["content"]

    ev_g = set()
    for r in ev:
        for d in docs_of(r["messages"][1]["content"]):
            ev_g |= grams(d)

    # 1. document pool
    text_of, key_of = {}, {}
    for r in rf:
        for d in docs_of(r["messages"][1]["content"]):
            k = norm(d)
            if k and k not in text_of:
                text_of[k] = d
    tainted = set()
    for k, d in text_of.items():
        g = grams(d)
        if g and len(g & ev_g) / len(g) >= TAINT:
            tainted.add(k)

    # 2. seeds
    seeds, drop = [], collections.Counter()
    for i, r in enumerate(rf):
        if not r["meta"].get("answerable"):
            continue
        tgt = r["messages"][2]["content"]
        if "##begin_quote##" not in tgt or "##end_quote##" not in tgt:
            drop["no_quote"] += 1; continue
        quote = tgt.split("##begin_quote##")[1].split("##end_quote##")[0].strip()
        ans = tgt.rsplit("##end_quote##", 1)[1].strip()
        ds = docs_of(r["messages"][1]["content"])
        gold = next((d for d in ds if quote[:60] and quote[:60] in d), None)
        if gold is None:
            drop["golden_not_locatable"] += 1; continue
        if norm(gold) in tainted:
            drop["golden_tainted"] += 1; continue
        if not ans:
            drop["empty_answer"] += 1; continue
        seeds.append({"raft_idx": i, "chunk_id": r["meta"].get("chunk_id"), "source": r["meta"].get("source"),
                      "split": split_of(str(r["meta"].get("chunk_id"))), "question": question_of(r["messages"][1]["content"]),
                      "golden": gold, "quote": quote, "answer": ans})

    # 3. home split of each golden document (first seed wins; conflicts recorded)
    home, conflicts = {}, 0
    for s in seeds:
        k = norm(s["golden"])
        if k in home and home[k] != s["split"]:
            conflicts += 1
        home.setdefault(k, s["split"])
    # seeds whose golden is also another split's golden would leak -> drop them
    seeds = [s for s in seeds if home[norm(s["golden"])] == s["split"]]

    # 4. IDF index over clean documents
    clean = [k for k in text_of if k not in tainted]
    df = collections.Counter()
    dwords = {}
    for k in clean:
        ws = set(words(text_of[k])); dwords[k] = ws
        df.update(ws)
    N = len(clean)
    idf = {w: math.log(N / (1 + c)) for w, c in df.items()}
    inv = collections.defaultdict(list)
    for k in clean:
        for w in dwords[k]:
            if df[w] < 0.05 * N:              # skip very common words for candidate generation
                inv[w].append(k)

    rng = random.Random(1337)
    items, short = [], collections.Counter()
    for s in seeds:
        qw = set(words(s["question"]))
        cand = collections.Counter()
        for w in qw:
            for k in inv.get(w, ()):
                cand[k] += idf[w]
        gk, na, qp = norm(s["golden"]), norm(s["answer"]), s["quote"][:60]
        picked = []
        for k, _ in cand.most_common(200):
            if k == gk or home.get(k, s["split"]) != s["split"]:
                continue
            t = text_of[k]
            if (qp and qp in t) or (len(na) >= 4 and na in norm(t)):
                continue
            picked.append(t)
            if len(picked) == 3:
                break
        if len(picked) < 3:
            short["not_enough_distractors"] += 1; continue
        if len(na) >= 4 and na in norm(" ".join(picked)):   # answer straddling two documents
            short["answer_across_docs"] += 1; continue
        a2 = [s["golden"], picked[0]]; rng.shuffle(a2)
        a3 = [s["golden"], picked[0], picked[1]]; rng.shuffle(a3)
        base = {"seed_raft_idx": s["raft_idx"], "chunk_id": s["chunk_id"], "source": s["source"],
                "split": s["split"]}
        for kind, docs, tgt in (("ans1", [s["golden"]], s["answer"]), ("ans2", a2, s["answer"]),
                                ("ans3", a3, s["answer"]), ("abs1", picked[:1], ABSTAIN),
                                ("abs2", picked[:2], ABSTAIN), ("abs3", picked[:3], ABSTAIN)):
            items.append({**base, "kind": kind, "id": f"{s['raft_idx']}:{kind}",
                          "messages": [{"role": "system", "content": system},
                                       {"role": "user", "content": render_user(docs, s["question"])},
                                       {"role": "assistant", "content": tgt}]})

    POOL.parent.mkdir(parents=True, exist_ok=True)
    with open(POOL, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")
    tab = collections.Counter((it["split"], it["kind"]) for it in items)
    man = {"rules": __doc__.split("Rules")[1].split("Outputs:")[0].strip(),
           "raft_sha256": hashlib.sha256(paths.RAFT.read_bytes()).hexdigest(),
           "eval_sha256": paths.EVAL_SHA256,
           "pool_sha256": hashlib.sha256(POOL.read_bytes()).hexdigest(),
           "documents": {"distinct": len(text_of), "tainted": len(tainted), "clean": len(clean)},
           "seed_drops": dict(drop), "home_conflicts_dropped": conflicts, "item_drops": dict(short),
           "n_seeds_used": len(items) // 6, "n_items": len(items),
           "table": {f"{a}|{b}": n for (a, b), n in sorted(tab.items())}}
    MANIFEST.write_text(json.dumps(man, indent=1))
    print(json.dumps({k: v for k, v in man.items() if k != "rules"}, indent=1))


def load(split: str, kinds=None) -> list[dict]:
    """split: 'edit_dev' or 'iter_val_0'..'iter_val_3'."""
    return [it for it in (json.loads(l) for l in open(POOL, encoding="utf-8"))
            if it["split"] == split and (kinds is None or it["kind"] in kinds)]


if __name__ == "__main__":
    main()

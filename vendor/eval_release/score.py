"""Reproducible scoring for the small-LM legal/financial QA benchmark.

Two reference-free metrics run offline (no API); the rubric-10 LLM judge is optional and needs an
OpenAI-compatible endpoint. The judge prompt is reproduced exactly from `judge_prompt.md`.

    # offline metrics only (token-F1 + fabrication rate)
    python score.py --predictions preds.jsonl

    # + rubric-10 LLM judge (OpenAI-compatible; also works with a base_url override)
    python score.py --predictions preds.jsonl --judge \
        --model <judge-model> --api-key <KEY> [--base-url <URL>] [--condition clean]

`preds.jsonl`: one JSON object per line, `{"id": "<eval id>", "answer": "<model output>"}`, with
ids from `eval_set.jsonl`. Prints aggregate metrics and writes `scored.jsonl`.
"""
from __future__ import annotations
import argparse, json, io, os, re
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "eval_set.jsonl")

RUBRIC = (
    'Score four dimensions, then nothing else:\n'
    '- "correctness" (0-5): factual agreement with the answer. 5 = fully right, 0 = wrong.\n'
    '- "completeness" (0-2): covers the key points, not just one of them.\n'
    '- "groundedness" (0-2): 2 = invents nothing; 0 = fabricated figures, cases or citations.\n'
    '- "clarity" (0-1): answers what was actually asked, without padding or contradiction.\n'
    '- "reason": one short sentence.\n'
    'The four add up to a score out of 10. Judge meaning, not wording.\n')


def words(s): return "".join(c if c.isalnum() or c.isspace() else " " for c in (s or "").lower()).split()
def token_f1(pred, ref):
    p, r = words(pred), words(ref)
    if not p or not r: return 0.0
    cc = {}
    for w in p: cc[w] = cc.get(w, 0) + 1
    ov = sum(min(c, r.count(w)) for w, c in cc.items())
    if not ov: return 0.0
    pr, rc = ov / len(p), ov / len(r); return 2 * pr * rc / (pr + rc)
def _nn(s): return s.replace(",", "").rstrip("%").rstrip(".")
def fabrication(pred, ctx):
    """1 if the answer states a number not supported by the context (reference-free proxy)."""
    cnorm = _nn((ctx or "").replace(",", ""))
    npd = {_nn(x) for x in re.findall(r"\d[\d,\.]*", pred or "")}
    ncx = {_nn(x) for x in re.findall(r"\d[\d,\.]*", ctx or "")}
    return float(len([x for x in npd if x and x not in ncx and x not in cnorm]) > 0)


def judge_prompt(system, context, question, reference, candidate):
    head = ("You are a strict evaluator of a question-answering system. Judge ONLY the "
            "CANDIDATE answer; you do not know which system produced it.\n")
    if reference:
        rule = ("The REFERENCE is a correct short answer. Grade the CANDIDATE against it. "
                "Judge meaning, not wording — a correct paraphrase scores high.\n")
        ref_block = f"\nREFERENCE:\n{reference}\n"
    else:
        rule = ("No reference answer is available. Grade the CANDIDATE on factual accuracy and "
                "whether it actually answers the question, using your own knowledge. Be strict: "
                "vague, evasive, repetitive or fabricated answers score low.\n")
        ref_block = ""
    ctx_block = f"\nCONTEXT:\n{context}\n" if context else ""
    ground = ("Groundedness is judged against the CONTEXT.\n" if context else "")
    body = (head + rule + "\n" + RUBRIC + ground + ctx_block + f"\nQUESTION:\n{question}\n"
            + ref_block + f"\nCANDIDATE:\n{candidate}")
    return body + ('\n\nReply ONLY as JSON: {"correctness":<0-5>,"completeness":<0-2>,'
                   '"groundedness":<0-2>,"clarity":<0-1>,"reason":"..."}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--condition", default=None, help="filter eval to one condition (e.g. clean)")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--api-key", default=os.environ.get("JUDGE_API_KEY"))
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    ev = {json.loads(l)["id"]: json.loads(l) for l in io.open(EVAL, encoding="utf-8") if l.strip()}
    preds = {json.loads(l)["id"]: json.loads(l)["answer"] for l in io.open(a.predictions, encoding="utf-8") if l.strip()}
    ids = [i for i in ev if i in preds and (a.condition is None or ev[i]["condition"] == a.condition)]
    print(f"scoring {len(ids)} items" + (f" (condition={a.condition})" if a.condition else ""))

    client = None
    if a.judge:
        from openai import OpenAI
        client = OpenAI(api_key=a.api_key, base_url=a.base_url) if a.base_url else OpenAI(api_key=a.api_key)

    def judge_one(i):
        e = ev[i]
        p = judge_prompt(e.get("system",""), e.get("context",""), e["question"], e.get("gold_answer",""), preds[i])
        for _ in range(3):
            try:
                r = client.chat.completions.create(model=a.model, temperature=0,
                        messages=[{"role": "user", "content": p}], response_format={"type": "json_object"})
                g = json.loads(r.choices[0].message.content or "{}")
                cc = max(0, min(5, int(g.get("correctness", 0)))); cp = max(0, min(2, int(g.get("completeness", 0))))
                gr = max(0, min(2, int(g.get("groundedness", 0)))); cl = max(0, min(1, int(g.get("clarity", 0))))
                return {"correctness": cc, "completeness": cp, "groundedness": gr, "clarity": cl,
                        "total": cc + cp + gr + cl, "grounded": gr == 2}
            except Exception:
                continue
        return None

    out, f1s, fabs, totals, grnd = [], [], [], [], []
    def work(i):
        e = ev[i]; row = {"id": i, "condition": e["condition"], "source": e.get("source"),
                          "token_f1": round(token_f1(preds[i], e.get("gold_answer","")), 4),
                          "fabrication": fabrication(preds[i], e.get("context",""))}
        if client is not None:
            j = judge_one(i)
            if j: row.update(j)
        return row
    with ThreadPoolExecutor(max_workers=a.workers if client else 1) as ex:
        out = list(ex.map(work, ids))
    for r in out:
        f1s.append(r["token_f1"]); fabs.append(r["fabrication"])
        if "total" in r: totals.append(r["total"]); grnd.append(1 if r["grounded"] else 0)
    io.open("scored.jsonl", "w", encoding="utf-8").write("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n")
    n = len(out)
    print(f"token-F1:      {sum(f1s)/n:.3f}")
    print(f"fabrication %: {100*sum(fabs)/n:.1f}")
    if totals:
        print(f"rubric total:  {sum(totals)/len(totals):.2f} / 10")
        print(f"grounded %:    {100*sum(grnd)/len(grnd):.1f}")
    print("wrote scored.jsonl")


if __name__ == "__main__":
    main()

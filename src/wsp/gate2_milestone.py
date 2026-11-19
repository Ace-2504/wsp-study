"""Milestone checks for Gate 2's selected arms (GPU): full held-out retention (all windows of both
disjoint sets; also validates the 512-window fast subset on a real edit) and the Phase-0 probes
(the approved verbatim-copy guardrail >= 0.93).

    python -m wsp.gate2_milestone --arms lora_lr0.0003 random_s0
"""
from __future__ import annotations

import argparse
import json

from . import checkpoints, engine, paths, probes, retention

OUT = paths.RUNS / "gate2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    a = ap.parse_args()
    checkpoints.verify_lock()
    base = engine.load_state("qa")
    _, tok, _ = checkpoints.load("qa")
    log = engine.RunLog("gate2")
    for arm in a.arms:
        delta = engine.load_delta(OUT / "deltas" / f"{arm}.safetensors")
        m = engine.materialize(base, delta)
        ret = retention.evaluate(m, OUT / "retention", arm)
        pr = probes.run(m, tok)
        (OUT / "probes").mkdir(parents=True, exist_ok=True)
        (OUT / "probes" / f"{arm}.json").write_text(json.dumps(pr))
        log.write("milestone", arm=arm, retention=ret,
                  probes={k: {kk: vv for kk, vv in v.items() if kk != "per_item"} for k, v in pr.items()})
        print(arm, "verbatim copy", pr["C_needle_verbatim"]["rate"], pr["C_needle_verbatim"]["ci95"], flush=True)
        del m


if __name__ == "__main__":
    main()

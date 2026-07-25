#!/usr/bin/env python
"""Statistics quoted in the report that analyze.py does not produce.

Everything recomputes from committed result files; no model runs.
- Paired jspace-vs-random McNemar tests (the J-space-specificity evidence
  for report section 4b; analyze.py only tests control vs jspace).
- MATH-500 accuracy by level and arm (section 4e).
- Arithmetic-probe accuracy by hop count and arm (section 4a).
- Reflective-marker rates in the transcript probe (section 5).

Usage: python scripts/extra_stats.py            # prints to stdout
       python scripts/extra_stats.py --out results/extra_stats.txt
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).parent.parent
ARMS = ("control", "jspace", "random")
BUDGETS = (0, 256, 512, 1024, 4096)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def mcnemar(rows: list[dict], ds: str, budget: int, a1: str, a2: str):
    """Exact two-sided binomial McNemar on paired (uid, sample_idx) outcomes."""
    per = collections.defaultdict(dict)
    for r in rows:
        if r["dataset"] == ds and r["budget"] == budget and r["arm"] in (a1, a2):
            per[(r["uid"], r["sample_idx"])][r["arm"]] = r["correct"]
    x = sum(1 for v in per.values() if v.get(a1) and not v.get(a2))
    y = sum(1 for v in per.values() if v.get(a2) and not v.get(a1))
    n = x + y
    p = sum(comb(n, i) for i in range(min(x, y) + 1)) * 2 / 2**n if n else 1.0
    return x, y, min(p, 1.0)


def main(out=sys.stdout):
    rows = load(ROOT / "results" / "main.jsonl")

    print("=== paired McNemar: jspace vs random ===", file=out)
    for ds in ("arith", "gsm8k", "math500", "aime24"):
        for b in BUDGETS:
            x, y, p = mcnemar(rows, ds, b, "jspace", "random")
            print(f"  {ds} budget={b}: jspace-only-right={x} "
                  f"random-only-right={y} p={p:.4f}", file=out)

    print("\n=== MATH-500 accuracy by level ===", file=out)
    acc = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r["dataset"] == "math500":
            k = (r["budget"], r["level"], r["arm"])
            acc[k][0] += r["correct"]
            acc[k][1] += 1
    for b in BUDGETS:
        for lvl in (1, 2, 3, 4, 5):
            cells = " ".join(
                f"{a}={acc[(b, lvl, a)][0] / acc[(b, lvl, a)][1]:.3f}" for a in ARMS
            )
            print(f"  budget={b} L{lvl}: {cells}", file=out)

    print("\n=== arith probe accuracy by hops ===", file=out)
    acc = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r["dataset"] == "arith":
            k = (r["budget"], r["level"], r["arm"])
            acc[k][0] += r["correct"]
            acc[k][1] += 1
    for b in BUDGETS:
        for hops in (1, 2, 3):
            cells = " ".join(
                f"{a}={acc[(b, hops, a)][0] / acc[(b, hops, a)][1]:.3f}" for a in ARMS
            )
            print(f"  budget={b} {hops}-hop: {cells}", file=out)

    print("\n=== reflection probe (budget 4096, GSM8K subset) ===", file=out)
    probe = load(ROOT / "results" / "reflection_probe.jsonl")
    pats = {
        "wait": r"\bwait\b",
        "double-check": r"double[- ]check",
        "verify": r"\bverify\b|\bverification\b",
        "alternatively": r"\balternatively\b|\banother way\b",
        "let-me-check": r"let me (re[- ]?)?check|let me confirm|make sure",
    }
    marks = collections.defaultdict(collections.Counter)
    toks = collections.Counter()
    n = collections.Counter()
    for r in probe:
        text = r["think_text"].lower()
        toks[r["arm"]] += r["n_think_tokens"]
        n[r["arm"]] += 1
        for name, pat in pats.items():
            marks[r["arm"]][name] += len(re.findall(pat, text))
    for arm in ("control", "jspace"):
        rates = " ".join(
            f"{k}={1000 * v / toks[arm]:.2f}" for k, v in sorted(marks[arm].items())
        )
        total = 1000 * sum(marks[arm].values()) / toks[arm]
        print(f"  {arm}: n={n[arm]} mean_think={toks[arm] / n[arm]:.0f} | "
              f"markers/1k tokens: {rates} | total={total:.2f}", file=out)
    per = collections.defaultdict(dict)
    for r in probe:
        per[r["uid"]][r["arm"]] = r["n_think_tokens"]
    ratios = [v["jspace"] / v["control"] for v in per.values()]
    print(f"  paired think-length ratio jspace/control: "
          f"median={statistics.median(ratios):.2f} (n={len(ratios)})", file=out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out:
        with open(args.out, "w") as f:
            main(f)
    else:
        main()

#!/usr/bin/env python
"""Aggregate results: accuracy tables with bootstrap CIs, paired McNemar
contrasts, rescue ratios, MATH-500 level breakdown, CoT-length stats.

Usage: python scripts/analyze.py results/main.jsonl [--out results/summary.csv]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load(path: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(path)]
    df = pd.DataFrame(rows)
    df["correct"] = df["correct"].astype(bool)
    return df


def bootstrap_ci(x: np.ndarray, n_boot: int = 5000, seed: int = 0):
    """95% bootstrap CI for a mean over problems."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (ds, arm, budget), g in df.groupby(["dataset", "arm", "budget"]):
        # average samples within problem first (AIME has 3 samples/problem)
        per_problem = g.groupby("uid")["correct"].mean().to_numpy()
        lo, hi = bootstrap_ci(per_problem)
        out.append({
            "dataset": ds, "arm": arm, "budget": budget,
            "n_problems": len(per_problem), "n_gen": len(g),
            "acc": per_problem.mean(), "ci_lo": lo, "ci_hi": hi,
            "extraction_fail_rate": g["extraction_failed"].mean(),
            "clip_rate": g["budget_clipped"].mean(),
            "mean_think_tokens": g["n_think_tokens"].mean(),
        })
    return pd.DataFrame(out).sort_values(["dataset", "budget", "arm"])


def mcnemar(df: pd.DataFrame, ds: str, budget: int, arm_a: str, arm_b: str):
    """Paired per-problem McNemar exact test between two arms."""
    sub = df[(df.dataset == ds) & (df.budget == budget)]
    a = sub[sub.arm == arm_a].groupby("uid")["correct"].mean() >= 0.5
    b = sub[sub.arm == arm_b].groupby("uid")["correct"].mean() >= 0.5
    common = a.index.intersection(b.index)
    a, b = a[common], b[common]
    n01 = int((~a & b).sum())  # a wrong, b right
    n10 = int((a & ~b).sum())
    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue * 1  # two-sided
    return {"n01": n01, "n10": n10, "p": float(p)}


def rescue_table(acc: pd.DataFrame) -> pd.DataFrame:
    """Relative accuracy retained under ablation: acc_arm / acc_control."""
    out = []
    for (ds, budget), g in acc.groupby(["dataset", "budget"]):
        ctrl = g[g.arm == "control"]["acc"]
        if ctrl.empty or ctrl.iloc[0] == 0:
            continue
        for arm in ("jspace", "random"):
            row = g[g.arm == arm]
            if row.empty:
                continue
            out.append({
                "dataset": ds, "budget": budget, "arm": arm,
                "retained": row["acc"].iloc[0] / ctrl.iloc[0],
                "abs_drop": ctrl.iloc[0] - row["acc"].iloc[0],
            })
    return pd.DataFrame(out)


def math_level_table(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df.dataset == "math500"].copy()
    if sub.empty:
        return pd.DataFrame()
    out = []
    for (level, arm, budget), g in sub.groupby(["level", "arm", "budget"]):
        out.append({"level": level, "arm": arm, "budget": budget,
                    "acc": g["correct"].mean(), "n": len(g)})
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="results/main.jsonl")
    ap.add_argument("--out", default="results/summary.csv")
    args = ap.parse_args()

    df = load(args.results)
    acc = accuracy_table(df)
    acc.to_csv(args.out, index=False)
    pd.set_option("display.width", 160)
    print("=== accuracy ===")
    print(acc.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== retained accuracy under ablation (acc_arm / acc_control) ===")
    rt = rescue_table(acc)
    print(rt.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    rt.to_csv(Path(args.out).with_name("rescue.csv"), index=False)
    ml = math_level_table(df)
    if not ml.empty:
        ml.to_csv(Path(args.out).with_name("math_levels.csv"), index=False)
    print("\n=== paired McNemar: control vs jspace ===")
    for ds in sorted(df.dataset.unique()):
        for budget in sorted(df.budget.unique()):
            r = mcnemar(df, ds, budget, "control", "jspace")
            print(f"  {ds} budget={budget}: ctrl-only-right={r['n10']} "
                  f"jspace-only-right={r['n01']} p={r['p']:.4f}")

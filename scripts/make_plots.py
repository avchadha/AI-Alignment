#!/usr/bin/env python
"""Report figures from main.jsonl.

Usage: python scripts/make_plots.py [results/main.jsonl] [--outdir report/figs]

Palette/marks follow the dataviz reference instance (first three categorical
slots in documented order; recessive grid; ordinal budget axis; CI bands).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARMS = ["control", "jspace", "random"]
ARM_LABEL = {"control": "No ablation", "jspace": "J-space ablation", "random": "Random matched-norm"}
COLOR = {"control": "#2a78d6", "jspace": "#008300", "random": "#e87ba4"}
SURFACE, INK, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
DS_ORDER = ["arith", "gsm8k", "math500", "aime24"]
DS_LABEL = {
    "arith": "Arithmetic probe (1-3 hops)",
    "gsm8k": "GSM8K (n=150)",
    "math500": "MATH-500 (n=150)",
    "aime24": "AIME 2024 (n=30, 3 samples)",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})


def load(path: str) -> pd.DataFrame:
    df = pd.DataFrame([json.loads(l) for l in open(path)])
    df["correct"] = df["correct"].astype(float)
    return df


def per_problem(g: pd.DataFrame) -> np.ndarray:
    return g.groupby("uid")["correct"].mean().to_numpy()


def boot_ci(x: np.ndarray, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def _budget_axis(ax, budgets):
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("thinking-token budget")


def fig_dose_response(df, outdir):
    budgets = sorted(df.budget.unique())
    ds_present = [d for d in DS_ORDER if d in set(df.dataset)]
    fig, axes = plt.subplots(1, len(ds_present), figsize=(3.1 * len(ds_present), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, ds_present):
        for arm in ARMS:
            ys, los, his = [], [], []
            for b in budgets:
                g = df[(df.dataset == ds) & (df.arm == arm) & (df.budget == b)]
                if g.empty:
                    ys.append(np.nan); los.append(np.nan); his.append(np.nan); continue
                x = per_problem(g)
                lo, hi = boot_ci(x)
                ys.append(x.mean()); los.append(lo); his.append(hi)
            xs = range(len(budgets))
            ax.fill_between(xs, los, his, color=COLOR[arm], alpha=0.15, linewidth=0)
            ax.plot(xs, ys, color=COLOR[arm], linewidth=2, marker="o", markersize=5,
                    label=ARM_LABEL[arm])
        _budget_axis(ax, budgets)
        ax.set_ylim(0, 1.02)
        ax.set_title(DS_LABEL.get(ds, ds), color=INK)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Accuracy vs. enforced thinking budget (95% bootstrap CI)", color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "fig1_dose_response.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_grouped_bars(df, outdir):
    budgets = sorted(df.budget.unique())
    ds_present = [d for d in DS_ORDER if d in set(df.dataset)]
    fig, axes = plt.subplots(1, len(ds_present), figsize=(3.4 * len(ds_present), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    w = 0.26
    for ax, ds in zip(axes, ds_present):
        for j, arm in enumerate(ARMS):
            xs, ys, errs = [], [], []
            for i, b in enumerate(budgets):
                g = df[(df.dataset == ds) & (df.arm == arm) & (df.budget == b)]
                if g.empty:
                    continue
                x = per_problem(g)
                lo, hi = boot_ci(x)
                xs.append(i + (j - 1) * (w + 0.02))
                ys.append(x.mean()); errs.append([x.mean() - lo, hi - x.mean()])
            ax.bar(xs, ys, width=w, color=COLOR[arm], label=ARM_LABEL[arm],
                   yerr=np.array(errs).T if errs else None,
                   error_kw=dict(ecolor=MUTED, lw=1, capsize=2))
        _budget_axis(ax, budgets)
        ax.set_ylim(0, 1.02)
        ax.set_title(DS_LABEL.get(ds, ds), color=INK)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Accuracy by ablation arm and thinking budget", color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "fig2_grouped_bars.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_difficulty(df, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    # (a) arith hops at budget 0
    ax = axes[0]
    sub = df[(df.dataset == "arith") & (df.budget == 0)]
    for arm in ARMS:
        ys = [sub[(sub.arm == arm) & (sub.level == h)]["correct"].mean() for h in (1, 2, 3)]
        ax.plot([1, 2, 3], ys, color=COLOR[arm], linewidth=2, marker="o", markersize=5,
                label=ARM_LABEL[arm])
    ax.set_xticks([1, 2, 3]); ax.set_xlabel("arithmetic hops"); ax.set_ylim(0, 1.02)
    ax.set_ylabel("accuracy"); ax.set_title("Arith probe, direct answer (budget 0)", color=INK)
    ax.legend(fontsize=8, loc="lower left")
    # (b) MATH-500 levels at budget 0
    ax = axes[1]
    sub = df[(df.dataset == "math500") & (df.budget == 0)]
    for arm in ARMS:
        ys = [sub[(sub.arm == arm) & (sub.level == lv)]["correct"].mean() for lv in range(1, 6)]
        ax.plot(range(1, 6), ys, color=COLOR[arm], linewidth=2, marker="o", markersize=5)
    ax.set_xticks(range(1, 6)); ax.set_xlabel("MATH-500 level"); ax.set_ylim(0, 1.02)
    ax.set_title("MATH-500, direct answer (budget 0)", color=INK)
    # (c) MATH-500 levels at max budget
    ax = axes[2]
    bmax = df.budget.max()
    sub = df[(df.dataset == "math500") & (df.budget == bmax)]
    for arm in ARMS:
        ys = [sub[(sub.arm == arm) & (sub.level == lv)]["correct"].mean() for lv in range(1, 6)]
        ax.plot(range(1, 6), ys, color=COLOR[arm], linewidth=2, marker="o", markersize=5)
    ax.set_xticks(range(1, 6)); ax.set_xlabel("MATH-500 level"); ax.set_ylim(0, 1.02)
    ax.set_title(f"MATH-500, thinking budget {bmax}", color=INK)
    fig.suptitle("Difficulty gradients", color=INK, y=1.03)
    fig.tight_layout()
    fig.savefig(outdir / "fig3_difficulty.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_rescue(df, outdir):
    budgets = sorted(df.budget.unique())
    ds_present = [d for d in DS_ORDER if d in set(df.dataset)]
    fig, axes = plt.subplots(1, len(ds_present), figsize=(3.1 * len(ds_present), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, ds_present):
        for arm in ("jspace", "random"):
            ys = []
            for b in budgets:
                c = per_problem(df[(df.dataset == ds) & (df.arm == "control") & (df.budget == b)])
                a = per_problem(df[(df.dataset == ds) & (df.arm == arm) & (df.budget == b)])
                ok = len(a) > 0 and len(c) > 0 and c.mean() > 0
                ys.append(a.mean() / c.mean() if ok else np.nan)
            ax.plot(range(len(budgets)), ys, color=COLOR[arm], linewidth=2, marker="o",
                    markersize=5, label=ARM_LABEL[arm])
        ax.axhline(1.0, color=BASE, linewidth=1, linestyle="--")
        _budget_axis(ax, budgets)
        ax.set_title(DS_LABEL.get(ds, ds), color=INK)
    axes[0].set_ylabel("accuracy retained\n(ablated / control)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Retained accuracy under ablation", color=INK, y=1.03)
    fig.tight_layout()
    fig.savefig(outdir / "fig4_rescue.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_cot_length(df, outdir):
    bmax = df.budget.max()
    ds_present = [d for d in DS_ORDER if d in set(df.dataset)]
    sub = df[df.budget == bmax]
    fig, axes = plt.subplots(1, len(ds_present), figsize=(3.1 * len(ds_present), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, ds_present):
        data = [sub[(sub.dataset == ds) & (sub.arm == arm)]["n_think_tokens"].to_numpy()
                for arm in ARMS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                        medianprops=dict(color=INK, linewidth=1.5))
        for patch, arm in zip(bp["boxes"], ARMS):
            patch.set_facecolor(COLOR[arm]); patch.set_alpha(0.6); patch.set_edgecolor(COLOR[arm])
        for el in ("whiskers", "caps"):
            for line in bp[el]:
                line.set_color(MUTED)
        ax.set_xticklabels(["control", "J-space", "random"], fontsize=8)
        ax.set_title(DS_LABEL.get(ds, ds), color=INK)
    axes[0].set_ylabel(f"thinking tokens used (budget {bmax})")
    fig.suptitle("Chain-of-thought length by ablation arm", color=INK, y=1.03)
    fig.tight_layout()
    fig.savefig(outdir / "fig5_cot_length.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="results/main.jsonl")
    ap.add_argument("--outdir", default="report/figs")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = load(args.results)
    fig_dose_response(df, outdir)
    fig_grouped_bars(df, outdir)
    fig_difficulty(df, outdir)
    fig_rescue(df, outdir)
    fig_cot_length(df, outdir)
    print(f"wrote 5 figures to {outdir}")

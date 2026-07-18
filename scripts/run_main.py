#!/usr/bin/env python
"""Main experiment: runs the frozen condition grid. Resumable (JSONL keyed).

Usage: python scripts/run_main.py --config configs/frozen.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jspace.data import select_problems
from jspace.runner import load_model_and_lens, run_conditions

ROOT = Path(__file__).parent.parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "frozen.yaml"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="subset of datasets to run (default: all in config)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model, tok, lens = load_model_and_lens(
        cfg["model_id"], cfg["lens_repo"], cfg["lens_file"], args.device
    )
    sel = select_problems(
        n_gsm8k=cfg["n_gsm8k"], n_math500=cfg["n_math500"], seed=cfg["problem_seed"]
    )
    datasets = args.datasets or cfg["datasets"]
    run_conditions(
        model, tok, lens,
        problems_by_ds={k: sel[k] for k in datasets},
        arms=cfg["arms"],
        budgets=cfg["budgets"],
        band=cfg["band"],
        out_path=ROOT / cfg["out_path"],
        batch_size=cfg["batch_size"],
        n_samples=cfg.get("n_samples", {}),
        answer_max_tokens=cfg["answer_max_tokens"],
        k=cfg["k"],
        base_seed=cfg["base_seed"],
        selection=cfg.get("selection", "cosine"),
    )

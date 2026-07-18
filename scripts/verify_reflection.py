#!/usr/bin/env python
"""Post-hoc probe: does J-ablation specifically remove reflective/hedging CoT?

Regenerates a small GSM8K subset (control vs jspace, budget 4096) saving the
FULL think text (main run stored only lengths), so the report's mechanism
claim can be checked against transcripts rather than asserted.

Usage: python scripts/verify_reflection.py --config configs/frozen.yaml
Writes: results/reflection_probe.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jspace.data import select_problems
from jspace.generate import SampleParams, build_prompt, generate_batch
from jspace.runner import load_model_and_lens, make_ablator

ROOT = Path(__file__).parent.parent
N_PROBLEMS = 20
BUDGET = 4096

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "frozen.yaml"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model, tok, lens = load_model_and_lens(
        cfg["model_id"], cfg["lens_repo"], cfg["lens_file"], args.device
    )
    sel = select_problems(
        n_gsm8k=cfg["n_gsm8k"], n_math500=cfg["n_math500"], seed=cfg["problem_seed"]
    )
    problems = sel["gsm8k"][:N_PROBLEMS]
    out_path = ROOT / "results" / "reflection_probe.jsonl"

    with out_path.open("w") as out:
        for arm in ("control", "jspace"):
            ablator = make_ablator(
                model, lens, arm, cfg["band"], k=cfg["k"], selection=cfg["selection"]
            )
            bs = cfg["batch_size"]
            for i in range(0, len(problems), bs):
                batch = problems[i : i + bs]
                prompts = [build_prompt(tok, p.question, direct=False) for p in batch]
                results = generate_batch(
                    model,
                    tok,
                    prompts,
                    ablator=ablator,
                    think_budget=BUDGET,
                    answer_max_tokens=cfg["answer_max_tokens"],
                    params=SampleParams(
                        seed=cfg["base_seed"]
                        + zlib.crc32(f"gsm8k|{arm}|{BUDGET}|{i}".encode()) % 10**6
                    ),
                )
                for p, r in zip(batch, results):
                    out.write(json.dumps({
                        "uid": p.uid,
                        "arm": arm,
                        "n_think_tokens": r.n_think_tokens,
                        "think_text": r.think_text,
                        "answer_text": r.answer_text,
                    }) + "\n")
                out.flush()
                print(f"[probe] arm={arm} {min(i + bs, len(problems))}/{len(problems)}")

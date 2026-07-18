#!/usr/bin/env python
"""Pilot: lens sanity readout, band calibration on held-out problems, SST-2
automatic-capability check. Run BEFORE freezing configs/frozen.yaml.

Usage: python scripts/run_pilot.py [--stage sanity|calibrate|sst2|all]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jspace.data import select_problems
from jspace.generate import SampleParams, generate_batch
from jspace.runner import BANDS, load_model_and_lens, make_ablator, run_conditions

MODEL_ID = "Qwen/Qwen3-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt"
RESULTS = Path(__file__).parent.parent / "results"


def stage_sanity(model, tok, lens):
    """Walkthrough-style readout: mid-layers should show interpretable tokens."""
    import jlens

    lm = jlens.from_hf(model, tok)
    prompt = "Fact: the currency used in the country shaped like a boot is the"
    lens_logits, model_logits, _ = lens.apply(lm, prompt, positions=[-1])
    print(f"prompt: {prompt!r}")
    for layer in sorted(lens_logits):
        top = lens_logits[layer][0].topk(5).indices
        print(f"  layer {layer:2d}: {[tok.decode([t]) for t in top]}")
    print("model top-5:", [tok.decode([t]) for t in model_logits[0].topk(5).indices])


def stage_calibrate(model, tok, lens, selection="cosine"):
    """Pilot problems through control (once) + jspace/random per candidate band."""
    sel = select_problems()
    pilot = {"pilot": sel["pilot"]}
    out = RESULTS / f"pilot2_{selection}.jsonl"
    run_conditions(
        model, tok, lens,
        problems_by_ds=pilot, arms=["control"], budgets=[0, 1024],
        band="medium",  # label only; control has no hooks
        out_path=out, batch_size=16, selection=selection,
    )
    for band in BANDS:
        run_conditions(
            model, tok, lens,
            problems_by_ds=pilot, arms=["jspace", "random"], budgets=[0, 1024],
            band=band, out_path=out, batch_size=16, selection=selection,
        )
    _summarize(out)


def stage_sst2(model, tok, lens, selection="cosine"):
    """Automatic-capability check: sentiment classification should survive
    J-space ablation (selectivity control, per the paper)."""
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/sst2", split="validation").select(range(100))
    out = RESULTS / f"pilot2_sst2_{selection}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["key"] for l in out.open()}
    with out.open("a") as f:
        for band in ["none", *BANDS]:
            arm = "control" if band == "none" else "jspace"
            ablator = None if band == "none" else make_ablator(model, lens, arm, band, selection=selection)
            correct = n = 0
            for i in range(0, len(ds), 10):
                rows = [ds[j] for j in range(i, min(i + 10, len(ds)))]
                keys = [f"sst2|{i+j}|{arm}|{band}|{selection}" for j in range(len(rows))]
                if all(k in done for k in keys):
                    continue
                prompts = [
                    tok.apply_chat_template(
                        [{"role": "user", "content":
                          f"Sentence: {r['sentence'].strip()}\n"
                          "Is the sentiment of this sentence positive or negative? "
                          "Answer with exactly one word: positive or negative."}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False,
                    )
                    for r in rows
                ]
                res = generate_batch(
                    model, tok, prompts, ablator=ablator, think_budget=None,
                    answer_max_tokens=8, params=SampleParams(seed=0),
                )
                for r, g, key in zip(rows, res, keys):
                    pred = g.answer_text.strip().lower()
                    ok = ("positive" in pred) == (r["label"] == 1) and (
                        ("positive" in pred) != ("negative" in pred)
                    )
                    correct += ok
                    n += 1
                    f.write(json.dumps({"key": key, "band": band, "arm": arm,
                                        "correct": bool(ok), "pred": pred[:40]}) + "\n")
            if n:
                print(f"SST-2 [{arm}/{band}/{selection}]: {correct}/{n} = {correct/n:.2%}")


def _summarize(path: Path):
    import collections

    acc = collections.defaultdict(lambda: [0, 0])
    for line in path.open():
        r = json.loads(line)
        key = (r["arm"], r["band"], r["budget"])
        acc[key][0] += r["correct"]
        acc[key][1] += 1
    print("\n=== pilot summary (arm, band, budget: acc) ===")
    for key in sorted(acc):
        c, n = acc[key]
        print(f"  {key}: {c}/{n} = {c/n:.2%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["sanity", "calibrate", "sst2", "all"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--selection", default="cosine", choices=["cosine", "logit"])
    args = ap.parse_args()

    model, tok, lens = load_model_and_lens(MODEL_ID, LENS_REPO, LENS_FILE, args.device)
    if args.stage in ("sanity", "all"):
        stage_sanity(model, tok, lens)
    if args.stage in ("calibrate", "all"):
        stage_calibrate(model, tok, lens, selection=args.selection)
    if args.stage in ("sst2", "all"):
        stage_sst2(model, tok, lens, selection=args.selection)

#!/usr/bin/env python
"""Diagnostic: how big is the J-space ablation delta relative to ||h||,
and what tokens get selected? Runs one math prompt through the ablated model."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from jspace.ablation import AblationConfig, JSpaceAblator
from jspace.generate import build_prompt
from jspace.runner import BANDS, load_model_and_lens

model, tok, lens = load_model_and_lens(
    "Qwen/Qwen3-4B", "neuronpedia/jacobian-lens",
    "qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt", "cuda")

band = BANDS["medium"]

prompt = build_prompt(tok, "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", direct=True)
ids = tok(prompt, return_tensors="pt").input_ids.cuda()

# capture h before/after per layer via manual _ablate calls on recorded h
from jlens.hooks import ActivationRecorder
layers_mod = model.model.layers
with torch.no_grad(), ActivationRecorder(layers_mod, at=band) as rec:
    model(ids)
for selection in ("cosine", "logit"):
    cfg = AblationConfig(band_layers=band, k=10, record_selection=True, selection=selection)
    abl = JSpaceAblator(model, lens.jacobians, cfg)
    print(f"\n=== selection={selection} ===")
    print(f"{'layer':>5} {'|delta|/|h| mean':>16} {'max':>6}   top tokens @ last pos")
    for l in band:
        h = rec.activations[l]
        h_new = abl._ablate(l, h)
        ratio = ((h - h_new).float().norm(dim=-1) / h.float().norm(dim=-1).clamp_min(1e-8))
        sel = abl.selected[l][0, -1].tolist()
        toks = [tok.decode([t]) for t in sel[:6]]
        print(f"{l:>5} {ratio.mean().item():>16.4f} {ratio.max().item():>6.3f}   {toks}")

"""Experiment orchestration: condition grid, checkpointed JSONL results.

Each result row is keyed by (dataset, uid, arm, band, budget, sample_idx); rows
already present in the output file are skipped, so interrupted runs resume.
"""

from __future__ import annotations

import json
import time
import zlib
from pathlib import Path

import torch

from .ablation import AblationConfig, JSpaceAblator
from .data import Problem, select_problems
from .evaluate import judge
from .generate import SampleParams, build_prompt, generate_batch

BANDS = {
    # 36 decoder blocks (indices 0-35). Workspace band per paper ~38-92% depth.
    "light": list(range(18, 27)),  # 50-75%
    "medium": list(range(14, 30)),  # 38-83%
    "heavy": list(range(14, 34)),  # 38-92%
}


def load_model_and_lens(model_id: str, lens_repo: str, lens_file: str, device: str = "cuda"):
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    model.eval()
    lens = jlens.JacobianLens.from_pretrained(lens_repo, filename=lens_file)
    return model, tok, lens


def make_ablator(
    model, lens, arm: str, band: str, k: int = 10, seed: int = 0, selection: str = "cosine"
):
    if arm == "control":
        return None
    layers = [l for l in BANDS[band] if l in lens.jacobians]
    if len(layers) < len(BANDS[band]):
        missing = sorted(set(BANDS[band]) - set(layers))
        print(f"[warn] lens missing jacobians for layers {missing}; using {layers}")
    cfg = AblationConfig(band_layers=layers, arm=arm, k=k, seed=seed, selection=selection)
    return JSpaceAblator(model, lens.jacobians, cfg)


def _existing_keys(path: Path) -> set[str]:
    keys = set()
    if path.exists():
        with path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    keys.add(r["key"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return keys


def _items_for(problems: list[Problem], n_samples: int):
    return [(p, s) for p in problems for s in range(n_samples)]


def run_conditions(
    model,
    tok,
    lens,
    *,
    problems_by_ds: dict[str, list[Problem]],
    arms: list[str],
    budgets: list[int],  # 0 => direct
    band: str,
    out_path: Path,
    batch_size: int = 8,
    n_samples: dict[str, int] | None = None,
    answer_max_tokens: int = 48,
    k: int = 10,
    base_seed: int = 0,
    selection: str = "cosine",
):
    n_samples = n_samples or {}
    done = _existing_keys(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ablators = {
        arm: make_ablator(model, lens, arm, band, k=k, selection=selection) for arm in arms
    }

    with out_path.open("a") as out:
        for ds, problems in problems_by_ds.items():
            items = _items_for(problems, n_samples.get(ds, 1))
            for arm in arms:
                for budget in budgets:
                    todo = [
                        (p, s)
                        for p, s in items
                        if f"{ds}|{p.uid}|{arm}|{band}|{budget}|{s}" not in done
                    ]
                    if not todo:
                        continue
                    print(f"[run] {ds} arm={arm} band={band} budget={budget}: {len(todo)} items")
                    for i in range(0, len(todo), batch_size):
                        batch = todo[i : i + batch_size]
                        t0 = time.time()
                        prompts = [
                            build_prompt(tok, p.question, direct=(budget == 0))
                            for p, _ in batch
                        ]
                        results = generate_batch(
                            model,
                            tok,
                            prompts,
                            ablator=ablators[arm],
                            think_budget=None if budget == 0 else budget,
                            answer_max_tokens=answer_max_tokens,
                            params=SampleParams(
                                # crc32: deterministic across processes (unlike hash())
                                seed=base_seed
                                + zlib.crc32(f"{ds}|{arm}|{budget}|{i}".encode()) % 10**6
                            ),
                        )
                        for (p, s), r in zip(batch, results):
                            # p.dataset (not the group key `ds`): pilot groups mix datasets
                            j = judge(p.dataset, r.answer_text, p.gold_answer)
                            row = {
                                "key": f"{ds}|{p.uid}|{arm}|{band}|{budget}|{s}",
                                "dataset": p.dataset,
                                "group": ds,
                                "uid": p.uid,
                                "level": p.level,
                                "arm": arm,
                                "band": band,
                                "selection": selection,
                                "budget": budget,
                                "sample_idx": s,
                                "correct": j.correct,
                                "extracted": j.extracted,
                                "extraction_failed": j.extraction_failed,
                                "n_think_tokens": r.n_think_tokens,
                                "n_answer_tokens": r.n_answer_tokens,
                                "budget_clipped": r.budget_clipped,
                                "finished": r.finished,
                                "gold": p.gold_answer,
                                "answer_text": r.answer_text[-2000:],
                                "think_len_chars": len(r.think_text),
                            }
                            out.write(json.dumps(row) + "\n")
                        out.flush()
                        dt = time.time() - t0
                        print(f"  batch {i//batch_size}: {dt:.1f}s ({dt/len(batch):.1f}s/item)")


def main_experiment(model, tok, lens, out_path: Path, band: str, **kw):
    sel = select_problems()
    run_conditions(
        model,
        tok,
        lens,
        problems_by_ds={k: sel[k] for k in ("gsm8k", "math500", "aime24")},
        arms=["control", "jspace", "random"],
        budgets=[0, 256, 1024, 4096],
        band=band,
        out_path=out_path,
        n_samples={"aime24": 3},
        **kw,
    )

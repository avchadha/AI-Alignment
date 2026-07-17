"""Dataset loading and canonical problem selection.

Problem sets are fixed by seed and shared across every (arm, budget) condition so
all comparisons are paired. Pilot problems are drawn first and excluded from the
main sets.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from datasets import load_dataset

GSM8K_ID = "openai/gsm8k"
MATH500_ID = "HuggingFaceH4/MATH-500"
# AIME source used throughout (assignment asks to state this exactly):
# HuggingFaceH4/aime_2024 = the 30 problems of AIME 2024 I and II.
AIME_ID = "HuggingFaceH4/aime_2024"

SEED = 20260716


@dataclass
class Problem:
    dataset: str
    uid: str  # stable id within dataset
    question: str
    gold_answer: str  # canonical final answer string
    level: int | None = None  # MATH-500 difficulty level 1-5, else None


def _gsm8k_gold(answer_field: str) -> str:
    return answer_field.split("####")[-1].strip().replace(",", "")


def load_gsm8k() -> list[Problem]:
    ds = load_dataset(GSM8K_ID, "main", split="test")
    return [
        Problem("gsm8k", f"gsm8k-{i}", row["question"], _gsm8k_gold(row["answer"]))
        for i, row in enumerate(ds)
    ]


def load_math500() -> list[Problem]:
    ds = load_dataset(MATH500_ID, split="test")
    return [
        Problem("math500", f"math500-{i}", row["problem"], row["answer"], level=int(row["level"]))
        for i, row in enumerate(ds)
    ]


def load_aime() -> list[Problem]:
    ds = load_dataset(AIME_ID, split="train")
    return [
        Problem("aime24", f"aime24-{row['id']}", row["problem"], str(row["answer"]))
        for row in ds
    ]


def select_problems(
    n_gsm8k: int = 150,
    n_math500: int = 150,
    n_pilot_gsm8k: int = 20,
    n_pilot_math500: int = 10,
    seed: int = SEED,
) -> dict[str, list[Problem]]:
    """Return {'pilot': [...], 'gsm8k': [...], 'math500': [...], 'aime24': [...]}.

    MATH-500 main set is stratified by level (n_math500/5 per level). Pilot
    problems are sampled first and excluded from main sets.
    """
    rng = random.Random(seed)

    gsm = load_gsm8k()
    gsm_idx = list(range(len(gsm)))
    rng.shuffle(gsm_idx)
    pilot_gsm = [gsm[i] for i in gsm_idx[:n_pilot_gsm8k]]
    main_gsm = [gsm[i] for i in gsm_idx[n_pilot_gsm8k : n_pilot_gsm8k + n_gsm8k]]

    math = load_math500()
    math_idx = list(range(len(math)))
    rng.shuffle(math_idx)
    pilot_math = [math[i] for i in math_idx[:n_pilot_math500]]
    pilot_ids = {p.uid for p in pilot_math}
    per_level = n_math500 // 5
    by_level: dict[int, list[Problem]] = {lv: [] for lv in range(1, 6)}
    for i in math_idx:
        p = math[i]
        if p.uid in pilot_ids:
            continue
        if p.level in by_level and len(by_level[p.level]) < per_level:
            by_level[p.level].append(p)
    main_math = [p for lv in range(1, 6) for p in by_level[lv]]

    return {
        "pilot": pilot_gsm + pilot_math,
        "gsm8k": main_gsm,
        "math500": main_math,
        "aime24": load_aime(),  # all 30, no subsampling
    }

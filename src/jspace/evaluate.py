"""Answer extraction and correctness judging.

Extraction failures are tracked separately from wrong answers: an ablated model
that emits garbage is a different failure mode from one that reasons to a wrong
number, and the analysis reports them separately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Judgement:
    extracted: str | None
    correct: bool
    extraction_failed: bool


def extract_boxed(text: str) -> str | None:
    """Return the contents of the last \\boxed{...} with balanced braces."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    i = start + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out).strip() if depth == 0 else None


_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_last_number(text: str) -> str | None:
    matches = _NUM_RE.findall(text)
    return matches[-1].replace(",", "") if matches else None


def _norm_numeric(s: str) -> float | None:
    try:
        return float(s.replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def judge(dataset: str, response: str, gold: str) -> Judgement:
    """Judge a model response against the gold answer.

    gsm8k/aime24 golds are integers/decimals -> numeric compare.
    math500 golds are LaTeX -> math_verify equivalence, numeric fallback.
    """
    extracted = extract_boxed(response)
    if extracted is None and dataset in ("gsm8k", "aime24"):
        extracted = extract_last_number(response)
    if extracted is None:
        return Judgement(None, False, True)

    if dataset in ("gsm8k", "aime24"):
        a, b = _norm_numeric(extracted), _norm_numeric(gold)
        return Judgement(extracted, a is not None and b is not None and abs(a - b) < 1e-6, False)

    # math500: symbolic equivalence
    try:
        from math_verify import parse, verify

        ok = bool(verify(parse(f"${gold}$"), parse(f"${extracted}$")))
    except Exception:
        ok = extracted.strip() == gold.strip()
    if not ok:
        a, b = _norm_numeric(extracted), _norm_numeric(gold)
        if a is not None and b is not None:
            ok = abs(a - b) < 1e-6
    return Judgement(extracted, ok, False)

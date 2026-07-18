import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jspace.evaluate import extract_boxed, extract_last_number, judge


def test_extract_boxed_balanced():
    assert extract_boxed(r"so \boxed{42}") == "42"
    assert extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_boxed(r"\boxed{a} then \boxed{b}") == "b"  # last one
    assert extract_boxed("no box") is None
    assert extract_boxed(r"\boxed{unclosed") == "unclosed"  # salvage


def test_extract_last_number():
    assert extract_last_number("answer is 1,234.5 dollars") == "1234.5"
    assert extract_last_number("x = -3") == "-3"
    assert extract_last_number("none") is None


def test_judge_gsm8k():
    assert judge("gsm8k", r"... \boxed{72}", "72").correct
    assert judge("gsm8k", "the answer is 72.", "72").correct  # fallback
    assert not judge("gsm8k", r"\boxed{71}", "72").correct
    j = judge("gsm8k", "I cannot solve this", "72")
    # fallback finds no number at all
    assert j.extraction_failed or not j.correct


def test_judge_math500_equivalence():
    assert judge("math500", r"\boxed{\frac{1}{2}}", r"\frac{1}{2}").correct
    assert judge("math500", r"\boxed{0.5}", r"\frac{1}{2}").correct
    assert not judge("math500", r"\boxed{2}", r"\frac{1}{2}").correct
    assert judge("math500", "garbage output", "5").extraction_failed


def test_judge_aime():
    assert judge("aime24", r"\boxed{204}", "204").correct
    assert not judge("aime24", r"\boxed{203}", "204").correct


def test_arith_probe_and_judging():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from jspace.data import make_arith_probe

    probs = make_arith_probe(n_per_hops=5)
    assert len(probs) == 15
    # deterministic across calls
    probs2 = make_arith_probe(n_per_hops=5)
    assert [p.question for p in probs] == [p.question for p in probs2]
    for p in probs:
        expr = p.question[len("What is "):-1]
        assert eval(expr.replace("*", "*")) == int(p.gold_answer)
    assert judge("arith", r"The final answer is \boxed{" + probs[0].gold_answer + "}",
                 probs[0].gold_answer).correct
    assert not judge("arith", r"\boxed{999999}", probs[0].gold_answer).correct

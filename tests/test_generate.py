import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jspace.ablation import AblationConfig, JSpaceAblator
from jspace.generate import SampleParams, build_prompt, generate_batch

QWEN_ID = "Qwen/Qwen3-4B"  # tokenizer only (small download); model is tiny+random


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(QWEN_ID)


@pytest.fixture(scope="module")
def tiny_model(tok):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=len(tok),  # real tokenizer ids must be valid
        max_position_embeddings=4096,
        head_dim=16,
    )
    return Qwen3ForCausalLM(cfg).eval()


def test_prompt_templates(tok):
    cot = build_prompt(tok, "What is 2+2?", direct=False)
    direct = build_prompt(tok, "What is 2+2?", direct=True)
    assert "step by step" in cot and "<think>" not in cot  # model opens think block
    assert "</think>" in direct  # enable_thinking=False closes it in the template
    assert "only the final answer" in direct


def test_budget_forcing_and_bookkeeping(tiny_model, tok):
    prompts = [build_prompt(tok, q, direct=False) for q in ["1+1?", "2+2?"]]
    res = generate_batch(
        tiny_model, tok, prompts,
        think_budget=8, answer_max_tokens=12,
        params=SampleParams(seed=0),
    )
    for r in res:
        # Random model never emits </think> on its own -> budget must clip.
        assert r.budget_clipped
        assert r.n_think_tokens <= 8
        assert r.n_answer_tokens <= 12
        assert "</think>" in tok.decode(r.token_ids)
        assert "The final answer is \\boxed{" in r.answer_text


def test_direct_arm_no_think(tiny_model, tok):
    prompts = [build_prompt(tok, "1+1?", direct=True)]
    res = generate_batch(
        tiny_model, tok, prompts, think_budget=None, answer_max_tokens=20,
        params=SampleParams(seed=0),
    )
    assert res[0].n_think_tokens == 0
    assert not res[0].budget_clipped
    assert res[0].answer_text.startswith("The final answer is \\boxed{")


def test_custom_loop_matches_hf_generate_greedy(tiny_model, tok):
    """Greedy decode via our two-cache-free path == HF generate (validates
    padding, position_ids, and cache handling)."""
    prompts = [build_prompt(tok, q, direct=True) for q in ["1+1?", "What is 10*3?"]]
    n_new = 12
    ours = generate_batch(
        tiny_model, tok, prompts, think_budget=None, answer_max_tokens=n_new,
        answer_prefix="",
        params=SampleParams(temperature=1e-6, top_p=1.0, top_k=1, seed=0),
    )
    enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left")
    hf = tiny_model.generate(
        **enc, do_sample=False, max_new_tokens=n_new,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    for i, r in enumerate(ours):
        hf_new = hf[i, enc.input_ids.shape[1]:].tolist()[: len(r.token_ids)]
        assert r.token_ids == hf_new[: len(r.token_ids)]


def test_ablated_generation_runs_end_to_end(tiny_model, tok):
    torch.manual_seed(1)
    d = tiny_model.config.hidden_size
    jac = {l: torch.randn(d, d) / d**0.5 for l in range(4)}
    abl = JSpaceAblator(tiny_model, jac, AblationConfig(band_layers=[1, 2], k=4))
    prompts = [build_prompt(tok, "1+1?", direct=False)]
    res = generate_batch(
        tiny_model, tok, prompts, ablator=abl, think_budget=6, answer_max_tokens=8,
        params=SampleParams(seed=0),
    )
    assert res[0].n_think_tokens <= 6 and res[0].n_answer_tokens <= 8

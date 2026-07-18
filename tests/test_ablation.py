import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jspace.ablation import AblationConfig, JSpaceAblator

D, VOCAB, LAYERS = 64, 512, 4


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(
        hidden_size=D,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=VOCAB,
        max_position_embeddings=1024,
        head_dim=16,
    )
    model = Qwen3ForCausalLM(cfg).eval()
    return model


@pytest.fixture(scope="module")
def jacobians():
    torch.manual_seed(1)
    return {l: torch.randn(D, D) / D**0.5 for l in range(LAYERS)}


def _vectors_from_ids(ablator, layer, top):
    """J-lens vectors for the token ids the ablator actually selected."""
    J = ablator._J[layer].float()
    u = ablator._u_eff.float()
    return u[top.reshape(-1)].reshape(*top.shape, -1) @ J  # [B, T, k, d]


def test_jspace_ablation_zeroes_selected_projections(tiny_model, jacobians):
    cfg = AblationConfig(band_layers=[1, 2], k=5, record_selection=True)
    abl = JSpaceAblator(tiny_model, jacobians, cfg)
    torch.manual_seed(2)
    h = torch.randn(2, 7, D)
    h_new = abl._ablate(1, h)
    v = _vectors_from_ids(abl, 1, abl.selected[1])
    # Projection of ablated residual onto each selected vector ~ 0.
    dots = torch.einsum("btkd,btd->btk", v, h_new.float())
    cos = dots / (v.norm(dim=-1) * h_new.float().norm(dim=-1, keepdim=True))
    assert cos.abs().max() < 1e-3
    # And the ablation actually changed the residual.
    assert (h_new - h).norm() > 0.01


def test_random_arm_matches_jspace_delta_norm(tiny_model, jacobians):
    torch.manual_seed(3)
    h = torch.randn(2, 5, D)
    abl_j = JSpaceAblator(tiny_model, jacobians, AblationConfig(band_layers=[1], k=5))
    abl_r = JSpaceAblator(
        tiny_model, jacobians, AblationConfig(band_layers=[1], k=5, arm="random", seed=7)
    )
    dj = (h - abl_j._ablate(1, h)).norm(dim=-1)
    dr = (h - abl_r._ablate(1, h)).norm(dim=-1)
    assert torch.allclose(dj, dr, rtol=2e-2, atol=1e-4)
    # But the actual perturbation directions differ.
    assert (abl_j._ablate(1, h) - abl_r._ablate(1, h)).norm() > 0.01


def test_exclusion_protects_tokens(tiny_model, jacobians):
    torch.manual_seed(4)
    h = torch.randn(1, 3, D)
    abl = JSpaceAblator(
        tiny_model, jacobians, AblationConfig(band_layers=[1], k=5, record_selection=True)
    )
    # Selection without exclusions (recorded by the ablator itself).
    abl._ablate(1, h)
    top = abl.selected[1]  # [1, 3, k]
    v = _vectors_from_ids(abl, 1, top)

    abl.set_exclusions(top)  # exclude exactly the would-be top-k
    h_excl = abl._ablate(1, h)
    abl.set_exclusions(None)
    h_plain = abl._ablate(1, h)
    # With the top-k excluded, their projections survive.
    dots_excl = torch.einsum("btkd,btd->btk", v, h_excl.float()).abs()
    dots_plain = torch.einsum("btkd,btd->btk", v, h_plain.float()).abs()
    assert (dots_excl.mean() > 10 * dots_plain.mean()).item()


def test_hooks_apply_in_forward(tiny_model, jacobians):
    cfg = AblationConfig(band_layers=[1, 2], k=5)
    abl = JSpaceAblator(tiny_model, jacobians, cfg)
    ids = torch.randint(0, VOCAB, (2, 6))
    with torch.no_grad():
        clean = tiny_model(ids).logits
        with abl:
            ablated = tiny_model(ids).logits
        clean2 = tiny_model(ids).logits  # hooks removed on exit
    assert not torch.allclose(clean, ablated)
    assert torch.allclose(clean, clean2)


def test_logit_selection_differs_and_zeroes(tiny_model, jacobians):
    torch.manual_seed(5)
    h = torch.randn(1, 4, D)
    abl_c = JSpaceAblator(
        tiny_model, jacobians, AblationConfig(band_layers=[1], k=5, record_selection=True)
    )
    abl_l = JSpaceAblator(
        tiny_model, jacobians,
        AblationConfig(band_layers=[1], k=5, selection="logit", record_selection=True),
    )
    out_c = abl_c._ablate(1, h)
    out_l = abl_l._ablate(1, h)
    assert not torch.equal(abl_c.selected[1], abl_l.selected[1])
    v = _vectors_from_ids(abl_l, 1, abl_l.selected[1])
    dots = torch.einsum("btkd,btd->btk", v, out_l.float())
    cos = dots / (v.norm(dim=-1) * out_l.float().norm(dim=-1, keepdim=True))
    assert cos.abs().max() < 1e-3
    assert not torch.allclose(out_c, out_l)

"""J-space ablation via forward hooks on decoder blocks.

Faithful to the procedure in "Verbalizable Representations Form a Global
Workspace in Language Models" (Transformer Circuits, 2026), section "J-space
ablation leaves most capabilities intact while impairing internal reasoning":

  At each token position, across a band of layers, identify the k=10 most
  strongly activated J-lens vectors (cosine similarity with the residual
  stream) and zero out the residual stream's projection onto them. Tokens in
  the top-10 of a clean forward pass at that position are never ablated.

Conventions follow the companion repo (anthropics/jacobian-lens): "layer l"
means the residual stream at the *output* of decoder block l (0-indexed), and
the unembedding applies the final norm then lm_head. We fold the final
RMSNorm's per-dim weight into the unembedding rows, so the J-lens vector for
vocab token c at layer l is

    v_c^l = J_l^T (g ⊙ u_c),

where u_c is the lm_head row and g the final-norm weight. Cosine-similarity
selection is invariant to the RMSNorm's per-position scalar, which we
therefore drop.

Arms:
  * "jspace": remove the joint (QR) projection of h onto the span of the
    selected top-k vectors.
  * "random": matched-norm control - remove a projection onto k fixed random
    orthonormal directions (per layer, seeded), rescaled so the per-position
    delta norm equals what the jspace arm would have removed there.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
from torch import nn

EPS = 1e-8


@dataclass
class AblationConfig:
    band_layers: list[int]  # decoder block indices whose outputs are ablated
    k: int = 10
    arm: str = "jspace"  # "jspace" | "random"
    selection: str = "cosine"  # "cosine" | "logit" (how top-k activation is ranked)
    exclude_top_m: int = 10  # clean-pass top-m output tokens never ablated
    pos_chunk: int = 64  # positions scored per chunk (memory control)
    seed: int = 0
    record_selection: bool = False  # keep selected token ids (analysis/tests)


def _block_output_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def _replace_block_output(output, new_hidden):
    if isinstance(output, tuple):
        return (new_hidden,) + tuple(output[1:])
    return new_hidden


class JSpaceAblator:
    """Context manager installing ablation hooks on a HF decoder model.

    Usage per forward pass:
        ablator.set_exclusions(ids)   # [B, T, m] clean-pass top-m token ids
        with ablator:                  # hooks active
            out = model(...)
    `set_exclusions(None)` disables the exclusion rule (used in unit tests).
    """

    def __init__(
        self,
        model: nn.Module,
        jacobians: dict[int, torch.Tensor],
        config: AblationConfig,
    ):
        self.model = model
        self.cfg = config
        layers = _find_decoder_layers(model)
        missing = [l for l in config.band_layers if l not in jacobians]
        if missing:
            raise ValueError(f"lens has no Jacobian for band layers {missing}")
        if max(config.band_layers) >= len(layers):
            raise ValueError("band layer index out of range")
        self._blocks = {l: layers[l] for l in config.band_layers}

        device = next(model.parameters()).device
        # Effective unembedding with final-norm weight folded in.
        u = _find_lm_head_weight(model).detach()
        g = _find_final_norm_weight(model).detach()
        self._u_eff = (u.float() * g.float()).to(device=device, dtype=torch.bfloat16)
        self._J = {
            l: jacobians[l].detach().to(device=device, dtype=torch.bfloat16)
            for l in config.band_layers
        }
        # Per-layer J-lens vector norms n_c = ||J_l^T u_eff_c||, chunked over vocab.
        self._vnorms: dict[int, torch.Tensor] = {}
        for l in config.band_layers:
            norms = torch.empty(self._u_eff.shape[0], device=device, dtype=torch.float32)
            for s in range(0, self._u_eff.shape[0], 8192):
                rows = self._u_eff[s : s + 8192] @ self._J[l]  # [chunk, d]
                norms[s : s + 8192] = rows.float().norm(dim=-1)
            self._vnorms[l] = norms.clamp_min(EPS)

        if config.arm == "random":
            d = self._u_eff.shape[1]
            gen = torch.Generator(device="cpu").manual_seed(config.seed)
            self._q_rand = {
                l: torch.linalg.qr(torch.randn(d, config.k, generator=gen))[0].to(
                    device=device, dtype=torch.float32
                )
                for l in config.band_layers
            }
        elif config.arm != "jspace":
            raise ValueError(f"unknown arm {config.arm!r}")

        self._exclusions: torch.Tensor | None = None
        self._handles: list = []
        self.selected: dict[int, torch.Tensor] = {}  # layer -> [B, T, k] ids

    def set_exclusions(self, ids: torch.Tensor | None):
        """ids: [B, T, m] token ids from the clean pass, aligned with the
        hidden states of the upcoming forward (T = positions being processed)."""
        self._exclusions = ids

    # -- hook machinery -------------------------------------------------

    def __enter__(self):
        for l, block in self._blocks.items():
            self._handles.append(block.register_forward_hook(self._make_hook(l)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, layer: int):
        def hook(module, args, output):
            h = _block_output_tensor(output)
            return _replace_block_output(output, self._ablate(layer, h))

        return hook

    # -- core math -------------------------------------------------------

    @torch.no_grad()
    def _ablate(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        """h: [B, T, d] residual at output of `layer`. Returns ablated h."""
        B, T, d = h.shape
        k = self.cfg.k
        J = self._J[layer]
        excl = self._exclusions
        if excl is not None and excl.shape[1] != T:
            raise RuntimeError(
                f"exclusion seq len {excl.shape[1]} != hidden seq len {T}"
            )

        out = h.clone()
        sel_chunks = []
        for s in range(0, T, self.cfg.pos_chunk):
            e = min(s + self.cfg.pos_chunk, T)
            hs = h[:, s:e, :]  # [B, C, d]
            hs_l = hs.to(J.dtype)
            t = hs_l @ J.T  # transported residual: J_l @ h, [B, C, d]
            scores = (t @ self._u_eff.T).float()  # lens logits, [B, C, V]
            if self.cfg.selection == "cosine":
                scores /= self._vnorms[layer]  # / ||v_c||
                scores /= hs.float().norm(dim=-1, keepdim=True).clamp_min(EPS)  # / ||h||
            elif self.cfg.selection != "logit":
                raise ValueError(f"unknown selection {self.cfg.selection!r}")
            if excl is not None:
                scores.scatter_(
                    -1,
                    excl[:, s:e, :].to(scores.device),
                    torch.finfo(scores.dtype).min,
                )
            top = scores.topk(k, dim=-1).indices  # [B, C, k]
            if self.cfg.record_selection:
                sel_chunks.append(top)

            # Gather selected J-lens vectors: v = J^T u_eff_c -> [B, C, k, d]
            u_sel = self._u_eff[top.reshape(-1)].reshape(B, e - s, k, -1)
            v = u_sel.float() @ J.float()  # [B, C, k, d]

            # Joint projection of h onto span(v) via QR.
            q, _ = torch.linalg.qr(v.transpose(-1, -2))  # [B, C, d, k]
            coeff = q.transpose(-1, -2) @ hs.float().unsqueeze(-1)  # [B, C, k, 1]
            delta = (q @ coeff).squeeze(-1)  # [B, C, d]

            if self.cfg.arm == "random":
                qr_ = self._q_rand[layer]  # [d, k]
                delta_r = (hs.float() @ qr_) @ qr_.T  # [B, C, d]
                scale = delta.norm(dim=-1, keepdim=True) / delta_r.norm(
                    dim=-1, keepdim=True
                ).clamp_min(EPS)
                delta = delta_r * scale

            out[:, s:e, :] = (hs.float() - delta).to(h.dtype)
        if self.cfg.record_selection:
            self.selected[layer] = torch.cat(sel_chunks, dim=1)
        return out


# -- HF architecture accessors (Qwen3 layout; matches jlens's Layout("model")) --


def _find_decoder_layers(model: nn.Module) -> nn.ModuleList:
    for path in ("model.layers", "transformer.h"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("could not locate decoder layers")


def _find_lm_head_weight(model: nn.Module) -> torch.Tensor:
    return model.get_output_embeddings().weight


def _find_final_norm_weight(model: nn.Module) -> torch.Tensor:
    for path in ("model.norm.weight", "transformer.ln_f.weight"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("could not locate final norm")


@contextlib.contextmanager
def maybe_ablate(ablator: JSpaceAblator | None):
    if ablator is None:
        yield
    else:
        with ablator:
            yield

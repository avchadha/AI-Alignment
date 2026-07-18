"""Batched generation with optional J-space ablation and thinking-budget forcing.

Decoding runs a custom loop with (optionally) two KV caches per batch:

  * a *clean* cache — no hooks — teacher-forced on the ablated continuation.
    Its top-m next-token predictions at each position give the exclusion set
    required by the paper ("we do not ablate any tokens that appear in the
    top-10 tokens of a clean forward pass").
  * an *ablated* cache — hooks active — which actually samples the output.

Budget forcing (thinking arms): rows that hit the thinking-token budget before
emitting </think> get "\n</think>\n\n" teacher-forced, then answer. The direct
arm (budget 0) uses enable_thinking=False chat templates and no thinking
machinery.

Answer forcing (all arms): on entering the answer phase, the constrained
prefix "The final answer is \\boxed{" is teacher-forced and only a short
completion is allowed. Without this, "direct answer" instructions leak chain
of thought into the answer text (the model writes step-by-step reasoning
despite being told not to), which would silently un-do the externalization
manipulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .ablation import JSpaceAblator

PREFILL_CHUNK = 128


@dataclass
class SampleParams:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    seed: int = 0


@dataclass
class GenResult:
    token_ids: list[int]
    text: str
    think_text: str
    answer_text: str
    n_think_tokens: int
    n_answer_tokens: int
    budget_clipped: bool
    finished: bool


@dataclass
class _RowState:
    phase: str  # "think" | "answer"
    n_think: int = 0
    n_answer: int = 0
    forced: list[int] = field(default_factory=list)  # queued teacher-forced ids
    clipped: bool = False
    done: bool = False
    tokens: list[int] = field(default_factory=list)


def _sample(logits: torch.Tensor, p: SampleParams, gen: torch.Generator) -> torch.Tensor:
    """logits: [B, V] -> sampled ids [B]."""
    logits = logits.float() / max(p.temperature, 1e-5)
    if p.top_k:
        kth = logits.topk(p.top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if p.top_p < 1.0:
        sorted_logits, idx = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        cum = probs.cumsum(dim=-1)
        mask = cum - probs > p.top_p  # keep tokens until top_p mass reached
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, idx, sorted_logits)
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, 1, generator=gen).squeeze(-1)


def _positions_from_mask(attn: torch.Tensor) -> torch.Tensor:
    return (attn.cumsum(-1) - 1).clamp_min(0)


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    *,
    ablator: JSpaceAblator | None = None,
    think_budget: int | None = None,  # None => direct arm (no think machinery)
    answer_max_tokens: int = 48,
    answer_prefix: str = "The final answer is \\boxed{",
    params: SampleParams = SampleParams(),
) -> list[GenResult]:
    device = next(model.parameters()).device
    exclude_m = ablator.cfg.exclude_top_m if ablator is not None else 0

    enc = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    input_ids = enc.input_ids.to(device)
    attn = enc.attention_mask.to(device)
    B, T0 = input_ids.shape

    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    force_close_ids = tokenizer.encode("\n</think>\n\n", add_special_tokens=False)
    prefix_ids = (
        tokenizer.encode(answer_prefix, add_special_tokens=False) if answer_prefix else []
    )
    nn_prefix_ids = (
        tokenizer.encode("\n\n" + answer_prefix, add_special_tokens=False)
        if answer_prefix
        else []
    )

    gen = torch.Generator(device=device).manual_seed(params.seed)
    states = [
        _RowState(phase="think" if think_budget is not None else "answer")
        for _ in range(B)
    ]
    if prefix_ids:
        for st in states:
            if st.phase == "answer":  # direct arm: constrain from the first token
                st.forced = list(prefix_ids)

    from transformers import DynamicCache

    abl_cache = DynamicCache()
    clean_cache = DynamicCache() if ablator is not None else None

    def forward(ids, mask, pos, cache, *, ablated: bool, exclusions=None):
        if ablated and ablator is not None:
            ablator.set_exclusions(exclusions)
            with ablator:
                out = model(
                    input_ids=ids, attention_mask=mask, position_ids=pos,
                    past_key_values=cache, use_cache=True,
                )
            ablator.set_exclusions(None)
        else:
            out = model(
                input_ids=ids, attention_mask=mask, position_ids=pos,
                past_key_values=cache, use_cache=True,
            )
        return out.logits

    # ---- prefill (chunked over positions to bound logits memory) ----
    positions = _positions_from_mask(attn)
    last_logits = None
    for s in range(0, T0, PREFILL_CHUNK):
        e = min(s + PREFILL_CHUNK, T0)
        ids_c, pos_c = input_ids[:, s:e], positions[:, s:e]
        mask_c = attn[:, :e]
        excl = None
        if ablator is not None:
            clean_logits = forward(ids_c, mask_c, pos_c, clean_cache, ablated=False)
            excl = clean_logits.topk(exclude_m, dim=-1).indices  # [B, chunk, m]
        last_logits = forward(ids_c, mask_c, pos_c, abl_cache, ablated=True, exclusions=excl)
    next_logits = last_logits[:, -1, :]

    # ---- decode loop ----
    cur_attn = attn
    max_steps = (think_budget or 0) + len(force_close_ids) + answer_max_tokens
    for _ in range(max_steps):
        # choose next token per row: forced queue > sampled
        sampled = _sample(next_logits, params, gen)
        next_ids = torch.empty(B, dtype=torch.long, device=device)
        for b, st in enumerate(states):
            if st.done:
                next_ids[b] = eos_id  # inert filler for finished rows
                continue
            if st.forced:
                tok = st.forced.pop(0)
            else:
                tok = int(sampled[b])
            next_ids[b] = tok
            st.tokens.append(tok)
            if st.phase == "think":
                st.n_think += 1
                if tok == think_end_id:
                    st.phase = "answer"
                    st.forced = list(nn_prefix_ids)
                elif st.n_think >= think_budget and not st.forced:
                    st.clipped = True
                    st.forced = list(force_close_ids) + list(prefix_ids)
                    st.phase = "answer"  # forced tokens complete the transition
            else:
                st.n_answer += 1
                if tok == eos_id or st.n_answer >= answer_max_tokens:
                    st.done = True
        if all(st.done for st in states):
            break

        step_ids = next_ids.unsqueeze(1)
        cur_attn = torch.cat([cur_attn, torch.ones(B, 1, dtype=attn.dtype, device=device)], dim=1)
        pos = _positions_from_mask(cur_attn)[:, -1:]
        excl = None
        if ablator is not None:
            clean_logits = forward(step_ids, cur_attn, pos, clean_cache, ablated=False)
            excl = clean_logits.topk(exclude_m, dim=-1).indices  # [B, 1, m]
        next_logits = forward(step_ids, cur_attn, pos, abl_cache, ablated=True, exclusions=excl)[:, -1, :]

    # ---- decode results ----
    results = []
    for st in states:
        toks = st.tokens
        # strip trailing filler after done (none appended post-done by design)
        text = tokenizer.decode(toks, skip_special_tokens=False)
        if think_end_id in toks:
            cut = toks.index(think_end_id)
            think_text = tokenizer.decode(toks[:cut], skip_special_tokens=True)
            answer_text = tokenizer.decode(toks[cut + 1 :], skip_special_tokens=True)
        else:
            think_text = ""
            answer_text = tokenizer.decode(toks, skip_special_tokens=True)
        results.append(
            GenResult(
                token_ids=toks,
                text=text,
                think_text=think_text,
                answer_text=answer_text,
                n_think_tokens=st.n_think,
                n_answer_tokens=st.n_answer,
                budget_clipped=st.clipped,
                finished=st.done,
            )
        )
    return results


def build_prompt(tokenizer, question: str, *, direct: bool) -> str:
    """Chat-template prompt. Direct arm disables thinking and forbids steps."""
    if direct:
        user = (
            f"{question}\n\nGive only the final answer within \\boxed{{}}. "
            "Do not write any intermediate steps, explanation, or reasoning."
        )
    else:
        user = (
            f"{question}\n\nPlease reason step by step, and put your final "
            "answer within \\boxed{}."
        )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=not direct,
    )

# J-space ablation vs. external chain of thought

Homework Zero (Fall 2026): does the internal–external "scratchpad" trade-off
between the J-space workspace and written chain of thought (Anthropic,
*Verbalizable Representations Form a Global Workspace in Language Models*,
Transformer Circuits, July 2026) continue to hold as math problems get harder?

**Model:** Qwen/Qwen3-4B ·
**Lens:** pre-fitted community Jacobian lens
[`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens)
(`qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt`, fitted with
the [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)
recipe on wikitext) ·
**Datasets:** GSM8K (`openai/gsm8k` test, n=150), MATH-500
(`HuggingFaceH4/MATH-500`, n=150 stratified 30/level), and **AIME 2024**
(`HuggingFaceH4/aime_2024` — the 30 problems of AIME 2024 exams I and II,
3 samples each).

## Design (3 ablation arms × 4 thinking budgets, fixed problem sets)

- **Arms:** `control` (no intervention); `jspace` — at every token position,
  across a band of layers, zero the residual stream's projection onto the
  k=10 most-active J-lens vectors (cosine similarity), never ablating tokens
  in the top-10 of a clean forward pass (paper-faithful); `random` —
  matched-norm control removing an equally large component along k fixed
  random directions in the same band (distinguishes J-space-specific effects
  from broad degradation).
- **External CoT axis:** enforced thinking-token budgets 0 (direct answer,
  `enable_thinking=False`) / 256 / 1024 / 4096, using Qwen3's native thinking
  mode; rows hitting the budget get `</think>` force-injected (compliance is
  enforced by the harness, not by instructions, since instruction-following
  itself degrades under ablation).
- **Pre-registration:** `report/hypothesis.md` and the frozen config are
  committed before the main runs (see git history). The ablation layer band
  is the only pilot-calibrated setting.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# or on a fresh GPU box: bash scripts/remote_bootstrap.sh

.venv/bin/pytest tests -q                       # unit tests (CPU, tiny model)
.venv/bin/python scripts/run_pilot.py           # lens sanity + band calibration + SST-2
# -> freeze band into configs/frozen.yaml, commit, then:
.venv/bin/python scripts/run_main.py --config configs/frozen.yaml
.venv/bin/python scripts/analyze.py results/main.jsonl
.venv/bin/python scripts/make_plots.py results/main.jsonl
```

Runs are resumable: results are appended to JSONL keyed by
`(dataset, problem, arm, band, budget, sample)` and existing keys are skipped.
Main experiment ≈ 4,680 generations (~2× forward cost in ablated arms for the
clean-pass exclusion rule); a few hours on one H100.

## Layout

```
src/jspace/ablation.py   # J-space ablation hooks (the core intervention)
src/jspace/generate.py   # two-cache batched decoding + budget forcing
src/jspace/data.py       # canonical problem selection (seeded, pilot held out)
src/jspace/evaluate.py   # boxed-answer extraction + math-verify judging
src/jspace/runner.py     # condition grid, checkpointed JSONL results
scripts/                 # pilot, main, analysis, plots, remote bootstrap
report/                  # hypothesis.md (pre-registered), report.md
```

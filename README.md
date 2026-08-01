# J-space ablation vs. external chain of thought

Homework Zero (Fall 2026): does the internal–external "scratchpad" trade-off
between the J-space workspace and written chain of thought (Anthropic,
*Verbalizable Representations Form a Global Workspace in Language Models*,
Transformer Circuits, July 2026) continue to hold as math problems get harder?

**Report:** [`report.pdf`](report.pdf) (repo root; source
[`report/report.tex`](report/report.tex), compile with
`tectonic -o . report/report.tex` or `latexmk -pdf` inside `report/`) ·
**Pre-registered hypothesis:** [`report/hypothesis.md`](report/hypothesis.md)
(committed before any runs; see git history)

**Model:** Qwen/Qwen3-4B ·
**Lens:** pre-fitted community Jacobian lens
[`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens)
(`qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt`, fitted with
the [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)
recipe on wikitext) ·
**Datasets:** GSM8K (`openai/gsm8k` test, n=150), MATH-500
(`HuggingFaceH4/MATH-500`, n=150 stratified 30/level), **AIME 2024**
(`HuggingFaceH4/aime_2024` — the 30 problems of AIME 2024 exams I and II,
3 samples each), and a synthetic 1/2/3-hop arithmetic probe (60 seeded
problems; keeps the direct-answer baseline off the floor).

## Design (3 ablation arms × 5 thinking budgets, fixed problem sets)

- **Arms:** `control` (no intervention); `jspace` — at every token position,
  across the frozen workspace layer band (medium = layers 14–29 of 36), zero
  the residual stream's projection onto the span of the k=10 most strongly
  activated J-lens vectors, ranked by raw lens logit (cosine ranking was also
  calibrated and agreed; both are implemented), never ablating tokens in the
  top-10 of a clean forward pass (paper-faithful); `random` — matched-norm
  control removing an equally large component along k fixed random directions
  in the same band (distinguishes J-space-specific effects from broad
  degradation).
- **External CoT axis:** enforced thinking-token budgets 0 / 256 / 512 /
  1024 / 4096 using Qwen3's native thinking mode. Compliance is enforced by
  the harness, not by instructions: generations exceeding the budget get
  `</think>` force-injected, and in **every** arm the answer segment is
  teacher-forced to `The final answer is \boxed{…}` with a 48-token cap
  (a pilot showed instruction-based "answer directly" conditions leak CoT
  into the answer text).
- **Pre-registration:** `report/hypothesis.md` and the frozen config
  (`configs/frozen.yaml`) are committed before the main runs (see git
  history). The layer band and selection rule are the only pilot-calibrated
  settings; the jspace-vs-random comparison was not a calibration criterion.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# or on a fresh GPU box: bash scripts/remote_bootstrap.sh

.venv/bin/pytest tests -q                       # unit tests (CPU, tiny model)
.venv/bin/python scripts/run_pilot.py           # lens sanity + band calibration + SST-2
# -> freeze band/selection into configs/frozen.yaml, commit, then:
.venv/bin/python scripts/run_main.py --config configs/frozen.yaml
.venv/bin/python scripts/analyze.py             # tables -> results/analysis_main.txt, CSVs
.venv/bin/python scripts/extra_stats.py --out results/extra_stats.txt
.venv/bin/python scripts/make_plots.py          # figures -> report/figs/
.venv/bin/python scripts/verify_reflection.py   # post-hoc CoT-content probe (report §5)
tectonic -o . report/report.tex                 # recompile the report PDF
```

Every table and figure in the report regenerates from the committed
results with no GPU (only `run_pilot.py`, `run_main.py`, and
`verify_reflection.py` need one):

- **Table 1** and the report's accuracy / retained-accuracy /
  control-vs-jspace McNemar numbers: `scripts/analyze.py` →
  `results/analysis_main.txt`, `results/summary.csv`.
- **Section 4's jspace-vs-random McNemar tests, MATH-500 level and
  arithmetic-hops breakdowns, and section 5's reflection-marker rates:**
  `scripts/extra_stats.py` → `results/extra_stats.txt`.
- **Figures 1–2 in the report (and repo figures 2, 3, 5):**
  `scripts/make_plots.py` → `report/figs/`.

Prompts are defined in `src/jspace/generate.py` (`build_prompt`, the forced
answer prefix, and the budget-forcing `</think>` injection).

Runs are resumable: results append to JSONL keyed by
`(dataset, problem, arm, band, budget, sample)` and existing keys are skipped.
Main experiment = 6,750 generations (3 arms × 5 budgets × 450 items; ~2×
forward cost in ablated arms for the clean-pass exclusion rule); ≈13.5 hours
on one H100. Completed outputs are committed: `results/main.jsonl` (full main
run), `results/analysis_main.txt` (accuracy/retained/McNemar tables),
`report/figs/` (figures 1–5), `logs/` (run logs).

## Layout

```
report.pdf               # the report (root, per submission requirements)
src/jspace/ablation.py   # J-space ablation hooks (the core intervention)
src/jspace/generate.py   # two-cache batched decoding + budget forcing + prompts
src/jspace/data.py       # canonical problem selection (seeded, pilot held out)
src/jspace/evaluate.py   # boxed-answer extraction + math-verify judging
src/jspace/runner.py     # condition grid, checkpointed JSONL results
scripts/                 # pilot, main, analysis, stats, plots, reflection probe
configs/frozen.yaml      # frozen main-run configuration (band, k, budgets, seeds)
report/                  # hypothesis.md (pre-registered), report.tex, figs/
results/                 # pilot + main JSONL, analysis tables, summary CSVs
logs/                    # pilot / calibration / main run logs
tests/                   # CPU unit tests for ablation, generation, judging
```

## External links

- Paper under investigation: [*Verbalizable Representations Form a Global
  Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/)
  (Anthropic, Transformer Circuits, July 2026;
  [arXiv:2607.15495](https://arxiv.org/abs/2607.15495))
- Lens weights (too large to commit; downloaded automatically by
  `run_pilot.py`/`run_main.py` via `jlens.JacobianLens.from_pretrained`):
  [`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens),
  file `qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt`
- Lens readout code / fitting recipe:
  [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)
  (installed from `requirements.txt`)
- Model: [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
- Datasets: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) ·
  [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) ·
  [HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)
  (all fetched automatically by `datasets`)

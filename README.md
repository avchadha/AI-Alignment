# J-space ablation vs. external chain of thought

Homework Zero (Fall 2026): does the internal–external "scratchpad" trade-off
between the J-space workspace and written chain of thought (Anthropic,
*Verbalizable Representations Form a Global Workspace in Language Models*,
Transformer Circuits, July 2026) continue to hold as math problems get harder?

**Report:** [`report/report.pdf`](report/report.pdf)
(source: [`report/report.md`](report/report.md)) ·
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
.venv/bin/python scripts/make_plots.py          # figures -> report/figs/
.venv/bin/python scripts/verify_reflection.py   # post-hoc CoT-content probe (report §5)
```

Runs are resumable: results append to JSONL keyed by
`(dataset, problem, arm, band, budget, sample)` and existing keys are skipped.
Main experiment = 6,750 generations (3 arms × 5 budgets × 450 items; ~2×
forward cost in ablated arms for the clean-pass exclusion rule); ≈13.5 hours
on one H100. Completed outputs are committed: `results/main.jsonl` (full main
run), `results/analysis_main.txt` (accuracy/retained/McNemar tables),
`report/figs/` (figures 1–5), `logs/` (run logs).

## Layout

```
src/jspace/ablation.py   # J-space ablation hooks (the core intervention)
src/jspace/generate.py   # two-cache batched decoding + budget forcing
src/jspace/data.py       # canonical problem selection (seeded, pilot held out)
src/jspace/evaluate.py   # boxed-answer extraction + math-verify judging
src/jspace/runner.py     # condition grid, checkpointed JSONL results
scripts/                 # pilot, main, analysis, plots, reflection probe, bootstrap
configs/frozen.yaml      # frozen main-run configuration (band, k, budgets, seeds)
report/                  # hypothesis.md (pre-registered), report.md, figs/
results/                 # pilot + main JSONL, analysis tables, summary CSVs
```

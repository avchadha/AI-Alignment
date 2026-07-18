# Does the J-space trade off against written chain of thought as problems get harder?

*Homework Zero — Fall 2026. Model: Qwen3-4B. Code: [repository link].*

## 1. Hypothesis

Recorded before the main experiments (`report/hypothesis.md`, committed
2026-07-17 before any runs; see git history): **ablating the J-space impairs
the multi-hop reasoning needed to solve math problems; the impairment grows
with problem difficulty; explicit chain of thought counteracts it, with the
rescue increasing with the amount externalized.** Concretely: (1) direct
answers suffer far more than CoT answers under J-space ablation, and far more
than under a matched-norm random ablation; (2) the control-vs-ablated gap
shrinks monotonically with an enforced thinking budget; (3) the relative loss
grows with difficulty (across GSM8K → MATH-500 → AIME, and within MATH-500
levels 1–5); (4) ablated models write longer CoT when allowed.

## 2. Experiment design

**Conditions.** 3 ablation arms × 5 enforced thinking budgets, fixed problem
sets shared by all conditions (paired comparisons).

- *Arms:* **control** (no intervention); **J-space ablation** — at every token
  position (prompt and generation), across the workspace layer band, zero the
  residual stream's projection onto the k=10 most strongly activated J-lens
  vectors, never ablating tokens in the top-10 of a clean forward pass (the
  paper's procedure); **random matched-norm** — remove a projection onto k
  fixed random directions in the same band, rescaled per position to remove
  exactly the norm the J-space ablation would have removed there. The random
  arm is the critical control: it distinguishes *J-space-specific* damage from
  *any equally large perturbation*.
- *External-CoT axis:* thinking budgets {0, 256, 512, 1024, 4096} tokens using
  Qwen3's native thinking mode. Compliance is enforced by the harness, not by
  instructions: rows exceeding the budget get `</think>` force-injected, and
  in **every** arm the answer segment is teacher-forced to
  `The final answer is \boxed{…}` with a 48-token cap. (A pilot showed that
  without this, "answer directly" instructions leak full chain of thought into
  the answer text, silently destroying the manipulation — see §3.)
- *Datasets:* GSM8K (test, n=150, seeded sample), MATH-500 (n=150, stratified
  30 per level 1–5), **AIME 2024** = `HuggingFaceH4/aime_2024` (the 30
  problems of the 2024 AIME I and II exams; 3 samples each), plus a synthetic
  **1/2/3-hop chained-arithmetic probe** (60 seeded problems) added after
  calibration revealed a floor effect (§3): it keeps the direct-answer
  baseline off the floor so an ablation effect on *internal* reasoning has
  room to appear.

**How either outcome stays visible.** If CoT does not rescue ablated
performance, the ablated curves stay flat as budget grows. If damage is
generic rather than J-specific, the random arm tracks the J-space arm. If
difficulty reverses the interaction, relative loss falls rather than rises
with hops/levels. Floor effects are flagged where they occur; wrong-answer
and unparseable-answer rates are tracked separately.

**Metrics.** Accuracy with 95% bootstrap CIs over problems (AIME samples
averaged within problem first); paired per-problem McNemar tests for key
contrasts; retained accuracy (ablated/control) per budget; CoT length
(`n_think_tokens`) as a measured outcome; budget-clip and extraction-failure
rates.

## 3. Experimental details

**Lens.** Pre-fitted community Jacobian lens for Qwen3-4B
(`neuronpedia/jacobian-lens`, fitted on wikitext with the
`anthropics/jacobian-lens` recipe; 479 prompts). Sanity readout reproduced the
paper's three-regime structure on the walkthrough example ("country shaped
like a boot" → mid-layers read 货币/currency/英镑/欧元/意大利, late layers
converge to " euro"), so the lens was accepted.

**Ablation implementation** (`src/jspace/ablation.py`; the official repo has
readout code only, so the intervention is implemented here). "Layer l" is the
residual at the output of decoder block l (jlens convention). The J-lens
vector of vocab token c is v_c = J_lᵀ(g ⊙ u_c) with the final RMSNorm weight
g folded into the unembedding row u_c. Activation ranking uses raw lens
logits (u_eff · J_l h); we also implemented cosine ranking — the paper's
wording admits either — and calibrated both (they agreed; logit was frozen
because it matches the lens's own readout and cosine promoted junk low-norm
tokens). The joint projection onto the span of the selected k vectors is
removed by QR. The clean-pass top-10 exclusion runs a second, hook-free model
pass with its own KV cache, teacher-forced on the ablated continuation (~2×
cost). Workspace band frozen to **medium = layers 14–29 of 36** (38–83%
depth) by pilot calibration; k=10.

**Calibration (30 held-out problems, never reused).** Under the constrained
protocol, control direct-answer accuracy is 16.7% vs 86.7% with 1024 thinking
tokens — the GSM8K/MATH direct baseline sits near floor for this 4B model.
J-space ablation at the medium band gives the largest relative direct-answer
drop (to 6.7%, both selection rules) while SST-2 sentiment classification
stays at 89–90% under *every* band and selection (= 89% control): the
ablation is selective in the paper's sense. The jspace-vs-random comparison
was deliberately not used as a calibration criterion.

**Runs.** 6,750 generations (3 arms × 5 budgets × 450 items) on one H100
(Lambda), HF transformers bf16, batch 16, Qwen3 recommended sampling
(thinking: T=0.6/top-p 0.95/top-k 20; non-thinking: T=0.7/top-p 0.8), fixed
seeds, resumable JSONL keyed by (dataset, problem, arm, band, budget,
sample). Judging: balanced-brace `\boxed{}` extraction (salvage for clipped
boxes), `math-verify` equivalence for MATH-500, numeric comparison otherwise.
Total compute ≈ 12 GPU-hours (~2× for ablated arms from the exclusion rule).
Reproduce: `scripts/run_pilot.py`, `scripts/run_main.py --config
configs/frozen.yaml`, `scripts/analyze.py`, `scripts/make_plots.py`.

## 4. Experimental results

*[PLACEHOLDER — final numbers, Figures 1–5, and the accuracy/rescue/McNemar
tables from scripts/analyze.py when the main run completes.]*

Interim signals already locked in from completed conditions:

- Arith probe, budget 0: control 100/75/25% by hops; J-space ablation
  100/55/5%; random 100/50/5%. One-hop computation is untouched; multi-hop
  collapses — but equally under both ablations.
- Dose-response: control GSM8K 15→38→75→90% across budgets 0→1024; ablated
  arith arms reach 100% by budget 512.

## 5. Analysis of results

*[PLACEHOLDER — written after full results. Structure: (a) which
pre-registered predictions held (multi-hop-specific impairment, one-step
survival, CoT rescue, dose-response); (b) which failed (J-space specificity:
random matched-norm ablation causes indistinguishable damage on this
model/lens); (c) floor effects on GSM8K/MATH/AIME direct answering and what
they imply about "internal reasoning" in small thinking-tuned models; (d)
alternative explanations — community lens quality (wikitext, 479 prompts),
selection-rule ambiguity, k/band severity, model scale; (e) what to test
next: paper-faithful lens fit on pretraining data, k and band sweeps,
per-position delta-norm matching audits, larger open models, and
patching/clamping instead of ablation.]*

# Does the J-space trade off against written chain of thought as problems get harder?

*Homework Zero — Fall 2026. Model: Qwen3-4B. Code: https://github.com/avchadha/AI-Alignment.*

## 1. Hypothesis

Recorded before the main experiments (`report/hypothesis.md`, committed
2026-07-16 before any runs; see git history): **ablating the J-space impairs
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
drop (to 6.7%; strictly largest under cosine ranking, tied with the heavy
band under logit) while SST-2 sentiment classification
stays at 89–90% under *every* band and selection (= 89% control): the
ablation is selective in the paper's sense. The jspace-vs-random comparison
was deliberately not used as a calibration criterion.

**Runs.** 6,750 generations (3 arms × 5 budgets × 450 items) on one H100
(Lambda), HF transformers bf16, batch 16, Qwen3 recommended sampling
(thinking: T=0.6/top-p 0.95/top-k 20; non-thinking: T=0.7/top-p 0.8), fixed
seeds, resumable JSONL keyed by (dataset, problem, arm, band, budget,
sample). Judging: balanced-brace `\boxed{}` extraction (salvage for clipped
boxes), `math-verify` equivalence for MATH-500, numeric comparison otherwise.
Total compute ≈ 13.5 GPU-hours for the main run (01:16–14:47 UTC; ~2× for
ablated arms from the exclusion rule) plus ~2 for pilots and calibration.
Reproduce: `scripts/run_pilot.py`, `scripts/run_main.py --config
configs/frozen.yaml`, `scripts/analyze.py`, `scripts/make_plots.py`.

## 4. Experimental results

All 6,750 generations completed (no extraction failures anywhere; clip rates
reported per condition). Full tables: `results/analysis_main.txt`; figures:
`report/figs/fig1`–`fig5`. Accuracy for the two headline datasets (95%
bootstrap CIs in Figure 1):

| budget | GSM8K ctrl | GSM8K J-abl | GSM8K rand | MATH ctrl | MATH J-abl | MATH rand |
|---|---|---|---|---|---|---|
| 0    | .153 | .140 | .100 | .227 | .220 | .180 |
| 256  | .380 | **.447** | .353 | .293 | **.347** | .273 |
| 512  | .753 | **.920** | .767 | .480 | **.573** | .493 |
| 1024 | .920 | .947 | .920 | .740 | .767 | .700 |
| 4096 | .967 | .960 | .973 | .873 | .847 | .833 |

**(a) No selective impairment at budget 0 (prediction 1 failed).** On GSM8K
and MATH-500 the direct-answer baseline sits at the floor calibration
predicted (15–23%), and J-ablation moves it by ~1 point (McNemar p = 0.80 /
1.00). On the arith probe, where the baseline has headroom (66.7%),
J-ablation does impair direct answering (53.3%, ctrl-only-right = 8 vs 0,
p = 0.0078) with a clean hop gradient — 1-hop untouched (100% → 100%), 2-hop
75% → 55%, 3-hop 25% → 5% — but the random matched-norm arm lands in the same
place (51.7%; 100/50/5% by hops; jspace-vs-random p = 1.0). Multi-hop-specific
damage, yes; J-space-specific, no.

**(b) A J-specific accuracy *gain* at intermediate budgets (the surprise).**
At budget 512 the J-ablated arm beats control by 16.7 points on GSM8K (92.0%
vs 75.3%; McNemar 27 vs 2, p < 10⁻⁵) and 9.3 points on MATH-500 (57.3% vs
48.0%; 18 vs 4, p = 0.0043), with the same sign at 256 on both datasets.
Unlike the impairments in (a), this effect **is** J-space-specific: the random
arm tracks control (76.7% / 49.3% at 512), and paired jspace-vs-random tests
give p < 10⁻⁴ (GSM8K 512), p = 0.024 (GSM8K 256), p = 0.017 (MATH-500 512).

**(c) The mechanism is shorter chain of thought (prediction 4 inverted).**
With the 4096 budget effectively unconstrained, J-ablated CoT is consistently
*shorter*: GSM8K 1,202 vs control's 1,703 mean thinking tokens (random:
1,642), arith 853 vs 1,049 (random: 1,149), MATH-500 2,326 vs 2,657
(Figure 5). Correspondingly the J-ablated arm is clipped less at every budget
where clipping bites (e.g. GSM8K@1024: 41% vs 69% clipped). The accuracy gain
in (b) lives exactly in the budgets where control is usually clipped mid-thought
and the denser J-ablated chains fit.

**(d) Dose–response and convergence (prediction 2, trivially).** Every arm
rises steeply with budget (Figure 1) and the arms converge by 1024–4096
(GSM8K 96–97%, MATH-500 83–87%, arith 98–100%; no significant contrasts).
CoT rescues the arith impairment completely — the ablated arms reach 100% by
budget 512 — so the gap does shrink with budget, but from a baseline whose
damage was never J-specific.

**(e) No difficulty interaction in the predicted direction (prediction 3
failed).** Within MATH-500 at budget 512 the J-ablation *advantage* appears
at levels 1–4 (+.10, +.17, +.10, +.10) and vanishes at level 5 (.23 both);
relative loss does not grow with difficulty at any budget (Figure 3). AIME
is floored below budget 4096 (0–4% everywhere), and at 4096 shows no
significant arm differences (control 33.3%, J-abl 36.7%, random 22.2%;
94% of AIME chains still clip at 4096, so it mostly measures clipping).

## 5. Analysis of results

**What held, what didn't.** Of the four pre-registered predictions, only the
weak form of prediction 2 (CoT rescue with dose–response) survives.
Prediction 1's selective impairment exists only where the direct-answer
baseline is off the floor (arith probe) and is matched by an equally large
random perturbation — on this model/lens, budget-0 damage is generic, not
J-specific. Prediction 3's difficulty interaction did not appear. Prediction 4
inverted: ablated models write *shorter*, not longer, chains.

**The result that matters.** The one robustly J-space-specific effect runs
opposite to the hypothesis: removing the top-k J-lens directions makes the
written chain of thought ~20–30% shorter at equal or better accuracy, which
shows up as a large accuracy gain wherever the thinking budget would otherwise
clip the control model mid-reasoning. A post-hoc transcript probe
(`scripts/verify_reflection.py`; 20 GSM8K problems regenerated at budget 4096
with full think-text saved, since the main run stored only lengths) shows the
omitted content is disproportionately *reflective*: J-ablated chains are not
just shorter (median paired length ratio 0.76) but carry fewer
reflection/hedging markers per thinking token — 6.7 vs 9.9 per 1k tokens
overall ("wait" 1.9 vs 3.2, "alternatively/another way" 1.6 vs 3.9,
"verify" 0.04 vs 0.25) — i.e. roughly half the double-checking loops and
alternative re-derivations per problem. This suggests the J-space directions
in this band
carry (or trigger) *self-monitoring/verbalization* content rather than the
load-bearing intermediate state the hypothesis assumed. If so, the J-space↔CoT
relationship in a small thinking-tuned model is not "internal workspace that
CoT can substitute for" but closer to "a knob on how much the model narrates";
the narration is partly redundant, since removing it costs nothing at
unconstrained budgets.

**Floor effects.** Truly forbidding externalized reasoning (teacher-forced
answer segment) drops un-ablated Qwen3-4B to 15–23% on GSM8K/MATH-500 and 0%
on AIME. A model this size has little "internal reasoning" to ablate on these
datasets — a core reason prediction 1 was untestable outside the arith probe,
and a caution for reading the paper's headline result onto small
thinking-tuned models.

**Alternative explanations to rule out.** (i) Lens quality: the community
lens (wikitext, 479 prompts) may fit a shallower subspace than the paper's;
a paper-faithful lens on pretraining data could restore J-specific
impairment. (ii) Severity coupling: although the random arm is norm-matched
per position, it is not matched for *where* in the residual basis the norm is
removed; a per-position delta-norm audit plus k/band sweeps would show whether
the budget-0 equivalence is an artifact of overall severity. (iii) Selection
rule: logit vs cosine ranking agreed in calibration, but both inherit the
lens; (iv) sampling: thinking-mode temperature 0.6 adds variance that McNemar
absorbs only per-sample. None of these plausibly explains away the mid-budget
gain, which replicates across two datasets with the random arm flat.

**Next steps.** Fit a paper-faithful lens; sweep k and the layer band;
delta-norm audits; test whether the CoT-shortening direction can be isolated
(clamp/patch single J-vectors rather than ablate the top-k span); check
whether shortened chains lose faithfulness (do stated steps still support the
answer?); and repeat on a larger open model where the direct-answer baseline
is off the floor without a synthetic probe.

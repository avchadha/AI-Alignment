# Does the J-space trade off against written chain of thought as problems get harder?

*Homework Zero — Fall 2026. Model: Qwen3-4B. Code: https://github.com/avchadha/AI-Alignment.*

## 1. Hypothesis

Recorded and committed before any main runs (`report/hypothesis.md`; see git
history): **ablating the J-space impairs the multi-hop reasoning needed to
solve mathematics problems; the impairment grows with problem difficulty;
and explicit chain of thought (CoT) counteracts it, increasingly with the
amount externalized.** Predictions: (1) direct answers suffer more from
J-space ablation than CoT answers, and far more than from a matched-norm
random ablation; (2) the control–ablated gap shrinks monotonically with
thinking budget; (3) relative loss grows with difficulty (GSM8K → MATH-500 →
AIME; MATH-500 levels 1–5); (4) ablated models write longer CoT when
permitted.

## 2. Experiment design

Three ablation arms × five thinking budgets on fixed problem sets (paired
comparisons). **Arms:** *control*; *J-space ablation* — at every token
position across the workspace layer band, remove the residual stream's
projection onto the span of the k = 10 most strongly activated J-lens
vectors, never ablating a clean pass's top-10 tokens (the paper's
procedure); *random matched-norm* — k fixed random directions in the same
band, rescaled per position to remove exactly the norm the J-space ablation
would remove; this arm separates J-space-specific damage from generic
degradation. **CoT axis:** budgets {0, 256, 512, 1024, 4096}
thinking tokens (Qwen3's native thinking mode), enforced by the harness:
over-budget generations receive a forced `</think>`, and every arm's answer
segment is teacher-forced to `The final answer is \boxed{…}` (48-token cap;
instruction-only direct-answer conditions leaked CoT in the pilot). **Datasets:** GSM8K (`openai/gsm8k` test, n = 150, seeded), MATH-500
(`HuggingFaceH4/MATH-500`, n = 150, 30 per level), **AIME 2024** —
`HuggingFaceH4/aime_2024`, the 30 problems of the 2024 AIME I and II exams,
3 samples per problem — and a synthetic 1/2/3-hop arithmetic probe (60
seeded problems) added after calibration revealed a direct-answer floor
(§3). Each opposite outcome is visible by construction: no rescue leaves
ablated curves flat; generic damage makes the random arm track the J-space
arm; a reversed difficulty interaction shows relative loss falling.
**Metrics:** accuracy with 95% bootstrap confidence intervals (AIME samples
averaged within problem), paired McNemar tests, retained accuracy, CoT
length, and clip and extraction-failure rates.

## 3. Experimental details

The intervention uses a pre-fitted community Jacobian lens for Qwen3-4B
(`neuronpedia/jacobian-lens`, wikitext, official recipe),
accepted after it reproduced the paper's walkthrough readout. The ablation
is implemented here (`src/jspace/ablation.py`; the official repository has
readout code only): J-lens vectors fold the final RMSNorm into the
unembedding, the joint projection onto the selected span is removed by QR,
and the clean-pass exclusion runs a second hook-free pass (≈2× cost).
Ranking uses raw lens logits (cosine ranking was calibrated and agreed). On
30 held-out calibration problems, control direct-answer accuracy is 16.7%
versus 86.7% with 1,024 thinking tokens; the medium band (layers 14–29 of
36) was frozen for the largest direct-answer drop (to 6.7%; tied with heavy
under logit) with SST-2 intact under every band and rule (89–90%, vs 89%
control). The J-space-versus-random comparison was excluded from calibration
criteria. Main run: 6,750 generations (3 arms × 5 budgets × 450 items), one
H100, ≈13.5 GPU-hours; judging by balanced-brace `\boxed{}` extraction with
`math-verify` equivalence for MATH-500. Reproduction: `README.md`,
`configs/frozen.yaml`.

## 4. Experimental results

All 6,750 generations completed; no extraction failures. Full tables:
`results/analysis_main.txt`, `results/extra_stats.txt`; figures 1–5:
`report/figs/`.

![Figure 1: accuracy versus enforced thinking budget by arm, with 95%
bootstrap confidence intervals.](figs/fig1_dose_response.png)

| budget | GSM8K ctrl | GSM8K J-abl | GSM8K rand | MATH ctrl | MATH J-abl | MATH rand |
|---|---|---|---|---|---|---|
| 0    | .153 | .140 | .100 | .227 | .220 | .180 |
| 256  | .380 | **.447** | .353 | .293 | **.347** | .273 |
| 512  | .753 | **.920** | .767 | .480 | **.573** | .493 |
| 1024 | .920 | .947 | .920 | .740 | .767 | .700 |
| 4096 | .967 | .960 | .973 | .873 | .847 | .833 |

**(a) No selective impairment at budget 0 (prediction 1 not supported).**
GSM8K/MATH-500 direct answering sits at the predicted floor; J-ablation
moves it ~1 point (McNemar p = 0.80 / 1.00). On the arithmetic probe, whose
baseline has headroom (66.7%), J-ablation does impair direct answering
(53.3%; p = 0.0078) with a clean hop gradient — 1-hop unaffected, 2-hop
75% → 55%, 3-hop 25% → 5% — but the random arm lands in the same place
(51.7%; p = 1.0): multi-hop-specific damage, not J-space-specific.

**(b) A J-space-specific gain at intermediate budgets.** At budget 512 the
J-ablated arm exceeds control by 16.7 points on GSM8K (92.0% vs 75.3%;
McNemar 27 vs 2, p < 10⁻⁵) and 9.3 points on MATH-500 (57.3% vs 48.0%;
p = 0.0043), with the same sign at 256. Unlike (a), this is specific to the
J-space directions: the random arm tracks control (76.7% / 49.3% at 512),
and paired J-space-versus-random tests give p < 10⁻⁴ (GSM8K, 512),
p = 0.024 (GSM8K, 256), p = 0.017 (MATH-500, 512).

**(c) The mechanism is shorter CoT (prediction 4 inverted).** At the
effectively unconstrained 4,096 budget, J-ablated chains are shorter: 1,202
versus 1,703 mean thinking tokens on GSM8K (random: 1,642), 853 vs 1,049 on
the probe, 2,326 vs 2,657 on MATH-500. The J-ablated arm is clipped less
wherever clipping is common (GSM8K at 1024: 41% vs 69%), and the gain in (b)
concentrates where control chains are truncated mid-reasoning.

**(d) Dose–response (prediction 2, weak form).** All arms rise steeply and
converge by budgets 1024–4096 (GSM8K 96–97%; MATH-500 83–87%). CoT fully
rescues the arithmetic impairment (J-ablated 100% at budget 512; control
93.3%, random 98.3%) — but from damage that was never J-space-specific.

**(e) No predicted difficulty interaction (prediction 3 not supported).**
Within MATH-500 at budget 512 the J-ablation *advantage* appears at levels
1–4 (+.10, +.17, +.10, +.10) and vanishes at level 5 (.23 both). AIME floors
below 4096 (0–4%) with no significant differences at 4096 (33.3% / 36.7% /
22.2%); 89–94% of chains still clip there, so that
condition largely measures truncation.

## 5. Analysis of results

Only the weak form of prediction 2 (rescue by CoT, with dose–response)
survives: prediction 1's impairment appears only off the floor and is
matched by random perturbation; prediction 3's interaction is absent;
prediction 4 inverted. The one robustly J-space-specific effect is opposite
in direction to the hypothesis: removing the top-k J-lens directions makes
written CoT roughly 25% shorter at equal or better accuracy. A post-hoc
transcript probe (`scripts/verify_reflection.py`; 20 GSM8K problems
regenerated with full think-text) shows the omitted content is
disproportionately *reflective*: J-ablated chains carry fewer reflection
markers per thinking token (6.7 vs 9.9 per 1,000; "wait" 1.9 vs 3.2) —
roughly half the double-checking and re-derivation loops per problem. This
suggests the ablated directions carry self-monitoring or verbalization
content rather than load-bearing intermediate state: they modulate how much the
model narrates rather than serving as an internal scratchpad.
Qualifications: floor
effects (with externalization prevented, un-ablated Qwen3-4B scores 15–23%
on GSM8K/MATH-500 and 0% on AIME — little internal reasoning to ablate,
cautioning against extrapolation to small thinking-tuned models) and
alternatives:
the wikitext-fitted community lens may capture a shallower subspace, the
random arm matches removed norm but not its residual-basis location, and
sampling adds variance. None of these plausibly explains the
intermediate-budget gain, which replicates across two datasets with the
random arm flat. Next: a paper-faithful lens fit; k and band sweeps with
delta-norm audits; clamping single J-vectors to isolate the shortening
direction; faithfulness checks on shortened chains; a larger model with an
off-floor direct baseline.

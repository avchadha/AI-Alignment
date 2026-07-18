# Pre-registered hypothesis and expectations

*Recorded and committed before any pilot or main experimental runs (see git history).*

## Hypothesis

Ablating the J-space negatively impacts the type of multi-hop reasoning necessary to
adequately solve math problems. This negative impact is exacerbated by more difficult
problems. However, explicit chain of thought counteracts these impairments, with
restorative effects increasing as the amount of externalized reasoning increases.

Operationalized predictions, in decreasing order of confidence:

1. **Selective impairment.** J-space ablation reduces accuracy far more in the
   direct-answer condition (thinking budget = 0) than in the full-CoT condition
   (budget = 4096), on GSM8K. A matched-norm random-direction ablation harms both
   conditions much less. (This is a replication of the paper's headline result on a
   different model, and a validity check on our implementation.)
2. **Dose–response rescue.** The accuracy gap between the J-ablated and control arms
   shrinks monotonically as the enforced thinking budget increases (0 → 256 → 1024 →
   4096 tokens).
3. **Difficulty interaction.** The *relative* accuracy loss under J-ablation (1 −
   acc_ablated / acc_control), at a fixed thinking budget, grows with problem
   difficulty — across datasets (GSM8K → MATH-500 → AIME) and within MATH-500
   (levels 1 → 5). Intuition: harder problems need more intermediate state per step,
   so even externalized reasoning leans on the internal workspace to produce each
   written step.
4. **Compensation.** Under J-ablation with an unlimited budget, the model writes
   *longer* chains of thought than the un-ablated control on the same problems
   (externalizing what it can no longer hold internally).

## How the opposite results would be visible

- If CoT does **not** rescue ablated performance, the J-ablated arm's accuracy stays
  flat (or falls) as budget grows, and prediction 2's gap does not shrink.
- If the effect is **generic degradation** rather than J-space-specific, the
  matched-norm random arm shows comparable losses to the J-ablated arm.
- If difficulty **reverses** the interaction (e.g., because hard problems already fail
  for other reasons), relative loss will be flat or decreasing in difficulty; we also
  record floor effects explicitly (AIME direct-answer accuracy for a 4B model is
  likely near 0 even without ablation, which would make the ablation contrast
  uninformative there — we state this in advance).
- Prediction 4 fails visibly if ablated CoT lengths are equal or shorter.

## Protocol amendments (recorded after pilot, BEFORE main runs)

1. **Constrained answers everywhere.** The pilot exposed a leak: with
   `enable_thinking=False` plus a "no intermediate steps" instruction, the
   model still wrote step-by-step reasoning inside its answer text, so the
   budget-0 arm was not a no-externalization condition. Fix: in every arm the
   answer segment is teacher-forced to `The final answer is \boxed{` with a
   48-token cap; externalized reasoning can therefore exist only in the
   measured think segment. Pilot v1 data (`results/pilot_v1_leaky_direct.jsonl`)
   is retained but not used.
2. **Top-k selection rule ambiguity.** The paper says the k "most strongly
   activated" J-lens vectors are ablated. Cosine-similarity ranking selects
   junk low-norm tokens (code formatting artifacts) in this community lens;
   raw lens-logit ranking matches the lens's own readout. We calibrate both on
   the pilot set and freeze one. Choice criterion (stated in advance): the
   selection/band configuration with a substantial J-ablation drop on
   direct answers while SST-2 sentiment (automatic capability) stays intact.
   The jspace-vs-random specificity comparison is NOT used as a selection
   criterion — it remains an open test of the main experiment.
3. **Calibration budgets.** Band calibration uses budgets {0, 1024} (not
   {0, 4096}) for speed; the full budget axis appears only in the main run.
4. **Floor effect confirmed, arith probe added.** Calibration under the fixed
   protocol shows un-ablated Qwen3-4B gets only ~17% on pilot GSM8K/MATH when
   truly forced to answer directly (vs ~87% with 1024 thinking tokens): the
   internal-reasoning baseline is near floor on the assignment datasets, as
   anticipated as a risk for AIME. To give the internal side of the trade-off
   measurable headroom, we add a synthetic 1/2/3-hop chained-arithmetic probe
   (60 problems, seeded, `jspace.data.make_arith_probe`) where the direct
   baseline should be off-floor, plus budget 512 (the steep dose-response
   region is between 0 and 1024). Recorded before any main runs.

## Known risks stated in advance

- The pre-fitted community lens (`neuronpedia/jacobian-lens`, wikitext, 479 prompts)
  may be lower quality than the paper's; we sanity-check its readouts before use.
- Ablation severity (layer band) is calibrated on a small held-out pilot set
  (excluded from main runs) and frozen before the main experiment.
- Budget forcing truncates thinking; on hard problems the 4096 cap may still clip
  reasoning. We report the clipped fraction.

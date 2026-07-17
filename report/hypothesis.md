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

## Known risks stated in advance

- The pre-fitted community lens (`neuronpedia/jacobian-lens`, wikitext, 479 prompts)
  may be lower quality than the paper's; we sanity-check its readouts before use.
- Ablation severity (layer band) is calibrated on a small held-out pilot set
  (excluded from main runs) and frozen before the main experiment.
- Budget forcing truncates thinking; on hard problems the 4096 cap may still clip
  reasoning. We report the clipped fraction.

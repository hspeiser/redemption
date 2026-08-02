# VQ2 Gates 0-4 World-Model POC

## Question

Can a model learned from recorded VQ2 interaction produce a controller that is
materially faster than the protected reference on gates 0-4, while retaining
its reliability, when tested in the real simulator?

This is a falsifiable gate before investing in a full-course learned simulator.

## Existing baseline

Source campaign:
`D:\ai-gp\training\vq2_v93_v79_gate10_macro_awr\20260731_220141`

- 200 episodes
- 178/200 (89%) officially crossed gate 4
- 57,518 transitions while targeting gates 0-4
- gate-4 crossing time: 9.733 s median and 9.733 s best
- per-target transition counts: 16,579 / 13,438 / 5,993 / 9,123 / 12,385

The baseline is both hard and unusually repeatable. Beating it is a meaningful
test; merely reproducing it is not.

## Data protocol

1. Split by complete episode, never by individual transition, to prevent
   adjacent frames leaking across splits.
2. Freeze approximately 15% of existing episodes as a locked test set before
   fitting anything.
3. Train on every valid pre-reset transition, including misses and crashes.
   Exclude reset discontinuities and post-impact motion from dynamics targets.
4. Add a small structured-identification set around the protected controller:
   bounded segment speed, command lead, thrust, and lateral-line perturbations.
   These probes create measurable action diversity without random exploration.
5. Keep the final live A/B flights out of every refit until that A/B round ends.

## Model

Use an ensemble of hybrid models, not one unconstrained neural simulator:

- analytic rigid-body/rate-loop model for the known physics;
- learned residual for body-frame velocity, body rates, and attitude deltas;
- separate observation/localizer corruption model for drift, landmark drought,
  covariance, and relocalization snaps;
- collision/gate-plane outcome head;
- five independently seeded members to expose epistemic uncertainty.

Train at multiple horizons (1, 4, 8, 16, and 32 control steps). The policy may
only exploit states where ensemble disagreement remains below a calibrated
threshold.

## Offline acceptance tests

The model must pass all of these before controller optimization counts:

1. Held-out 0.13 s, 0.27 s, 0.53 s, and 1.07 s rollout error is reported in
   body position, velocity, attitude, and rates.
2. Predicted gate crossings and collisions agree with locked episodes.
3. Predicted changes caused by the structured probes have the correct sign and
   useful rank correlation with reality.
4. Ensemble intervals are calibrated: large real errors must coincide with
   large predicted uncertainty.
5. Replaying the protected controller reproduces its gate-4 time and clearance
   distribution rather than only one nominal trajectory.

## Candidate optimization

Optimize a bounded residual over the protected reference/controller. Search can
use CEM first and policy gradients second, but it must:

- remain inside the measured action/state support or pay a strong support cost;
- optimize worst-member or CVaR performance, not the ensemble mean alone;
- penalize ensemble disagreement;
- require official-order crossings of gates 0-4;
- retain clearance and localization visibility margins;
- start from the real at-rest, pitched-pad initial condition.

No hand-authored per-gate trim may be introduced after looking at candidate live
failures. A failed candidate can only improve by adding that round's data and
refitting the same pipeline.

## Live proof

Run the protected baseline and candidate interleaved (randomized ABBA order),
with the same localizer, countdown, controller wrapper, and safety logic. End an
episode immediately after the official gate-4 event.

- Calibration: 4 baseline + 4 candidate flights
- Final locked test: 10 baseline + 10 candidate flights
- Primary metric: median official gate-4 crossing time
- Guardrails: pass rate, p10 gate clearance, collision count, timing health,
  landmark age, and localization covariance

## Go/no-go thresholds

From the 9.733 s baseline:

- 5% faster: 9.247 s (interesting, but insufficient)
- 10% faster: 8.760 s (minimum technical proof)
- 15% faster: 8.273 s (strong reason to expand to the full course)
- 20% faster: 7.787 s (approximately full-lap sub-30-grade improvement)

The POC passes only if the candidate reaches gate 4 in at most 8.760 s median,
passes at least 9/10 locked flights, and does not materially regress clearance,
timing health, or localization. A 15% improvement is the preferred expansion
threshold. A 20% improvement would be direct evidence that the method has the
scale needed for sub-30, though early-course launch constraints may make this
segment harder to accelerate proportionally.

If the candidate wins offline but fails live, perform at most two Dyna rounds:
append the failed live data, refit, and re-optimize. If it still cannot clear the
minimum threshold, stop the full-course world-model effort and retain the data
and system-identification improvements.

## Why this is cheap

The current controller reaches gate 4 in under ten seconds. Even with the
official countdown and reset overhead, the locked 20-flight A/B requires only a
small fraction of a full 200-episode campaign. The simulator is used for proof
and targeted model correction, while the expensive optimization occurs offline.

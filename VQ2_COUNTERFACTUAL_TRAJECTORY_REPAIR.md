# VQ2 counterfactual trajectory repair

Updated 2026-08-01. Status: implemented and exercised as an optional
augmentation; not a promotion path by itself.

## First implementation result

The first gates 0-4 proof-of-concept used the healthy-timing gate-2 failure
from live session `vq2_20260801_212358`, episode 2.  The immutable upstream
snapshot artifact is
`D:\ai-gp\repairs\g2_ep2_20260801_snapshot_v2`.

- Original-action reproduction passed at the 1.5-second rollback: the same
  gate-2 failure occurred in 42.2-46.9% of perturbed worlds across world-model
  versions v23, v24, and v25.
- Actor-only bounded repair search failed (about 14.5% success), identifying
  insufficient residual authority rather than inventing a candidate.
- A joint gate-1-to-2 search found a locally robust structural repair: pooled
  fresh-audit success 97.28%, no model family below 96.73%, and 0.311 m p10
  clearance.  It required a +0.35 m gate-1 crossing offset, a -0.091 m
  vertical offset, and a 1.056 velocity multiplier in addition to bounded
  residual actions.
- Synthetic actor rows were exported with `synthetic_repair=true`, actor
  weight 0.25, critic weight 0, and `dynamics_eligible=false`.
- Distilled actors were rejected by the full gates 0-4 regression.  The best
  state-gated version scored about 27.6% under impulses versus 30.7% for the
  structural geometry alone and 33.3% for the protected source across the
  three model families.  It was not flown live.

The result is therefore useful diagnosis, not a promotable policy.  This
failure needs evidence from repeated independent live episodes before the
shared reference line is changed.  The implementation deliberately preserves
the protected champion and keeps the rejected actor quarantined.

## Second implementation result: aggressive gates 0-4 gate-3 misses

The 2026-08-02 integration pass used two independent healthy-timing gate-3
collisions from live session `vq2_g0g4_geometry_live/20260801_205948`
(episodes 0 and 1), with episode 3 as the matching 8.778-second success.

- Branch restoration was corrected to keep the recorded EKF belief fixed
  while sampling a separate latent physical position.  The previous helper
  perturbed physical state and exposed that sampled truth to the controller,
  eliminating the localization error a repair must survive.
- Branch position uncertainty is now explicitly calibrated and recorded in
  the reproduction report; repair search inherits the exact same floor and
  scale.  A measured 0.10 m floor made both failures reproducible at the
  0.4-second rollback without unrealistically making longer horizons fail in
  every world.
- A rollout-fine-tuned v24 dynamics ensemble reduced held-out 32-step position
  p90 from 0.535 m to 0.497 m.  It did not by itself make the real failure
  reproducible at useful horizons, confirming that belief/physical separation
  and uncertainty calibration were also required.
- The bounded 0.4-second action-only repair was rejected; it reached roughly
  70-75% local success, below the 95% acceptance floor.
- The final action-plus-geometry search was also rejected; its best local
  success remained roughly 76-78% despite allowing up to 0.25 m lateral,
  0.20 m vertical, and 10% speed adjustment.

No synthetic rows or actor checkpoints from these searches were promoted.
The measured conclusion is that this failure needs an earlier reference or
controller intervention, not a last-moment counterfactual actor correction.

The first earlier-intervention check changed only gate 3's reference lateral
offset from -0.163 m to -0.350 m and audited it against the unchanged schedule
on 2,048 paired randomized worlds per model (v22, v23, and rollout-tuned v24).
It improved tier success by 1.18-1.76 percentage points on v23/v24, but only
0.39 points on v22 while slightly reducing v22 finish rate.  Worst-model tier
success was 61.96%, far below the live-promotion floor.  The variant is
therefore quarantined as diagnosis, not a flight candidate.  A shared
reference repair still requires either a stronger multi-parameter result or
more repeated real failures that identify the correct earlier intervention.

A subsequent gate-3-only CEM search jointly varied turn lead, thrust, speed,
trajectory blend, lateral/vertical crossing offsets, and rate scale while
freezing every other gate.  It optimized the worst score across v22/v23/v24,
with impulses and live-estimator realism enabled.  The fresh-world selector
chose an extreme -0.372 m lateral / +0.400 m vertical candidate.  Its locked
4,096-world audit reached only 61.08% worst-model tier success and 71.07%
worst-model finish rate.  Because it both touched a geometry boundary and
missed the promotion floor by a wide margin, it was rejected without a live
probe.  This closes the current gate-3 repair attempt: counterfactual repair
remains available for future repeated failure classes, but it is not the
primary optimizer for the present gates 0-4 campaign.

## Summary

Turn every useful live failure into a counterfactual search problem:

1. restore one or more recorded states shortly before the failure;
2. replay the original controls to verify that the world-model ensemble can
   approximately reproduce the failure;
3. search thousands of bounded residual-action sequences from those states;
4. retain only repairs that pass the intended gate robustly across model
   members, localization perturbations, and nearby initial states;
5. use accepted repairs as synthetic supervision for the actor, critic,
   reference trajectory, replay sampler, and failure-analysis tools;
6. validate the resulting complete closed-loop policy in the real simulator.

This is counterfactual trajectory repair, not simulator rollback. The real VQ2
simulator still resets to the beginning. All branching happens offline in the
learned/analytic fast world model, preferably on Gipsydanger.

## Motivation

A failed live episode currently tells the learner that an action sequence was
bad, but usually does not reveal which earlier decision would have saved it.
With a short-horizon world model, one expensive failure can instead generate
thousands of controlled alternatives around the exact state the policy
actually visited.

At 8-10 m/s, a 0.2-second rollback is only six control steps and approximately
1.6-2.0 m of travel. That may be sufficient for a small trim but too late for
a major turn. Every failure should therefore branch from multiple horizons,
initially:

- 0.2 s before failure: six steps at 30 Hz;
- 0.4 s before failure: twelve steps;
- 0.7 s before failure: twenty-one steps;
- 1.0 s before failure: thirty steps;
- optional earlier branch at the previous official gate event.

The earliest rollback that produces a robust, low-authority repair is
preferred. If no rollback can repair the failure inside measured support, the
system records that the current reference/controller needs a larger structural
change or additional real identification data.

## Scope

The first implementation targets physical gate misses and collisions with
healthy timing and usable localization. It can later expand to:

- wrong-side or wrong-order gate approaches;
- gate timeouts and stagnation;
- overspeed failures;
- localization drought recovery;
- poor next-gate entry state despite a successful current crossing;
- deterministic near-misses with insufficient safety margin.

Simulator-clock stalls, stale packet failures, reset discontinuities, and
post-impact falling are infrastructure labels, not trajectory-repair targets.

## Required inputs

Every repair job must reference immutable real data:

- source session and episode;
- source configuration, map, detector, localizer, controller, and checkpoint
  hashes;
- timestamped localizer state and covariance;
- position, velocity, attitude, and body rates;
- current and future gate geometry;
- previous canonical and wire commands;
- official gate and collision events;
- timing-health and landmark-age history;
- complete original action sequence through the terminal event;
- world-model ensemble and calibration-report hashes.

The canonical action order in the current implementation is:

```text
[roll-rate command, pitch-rate command, yaw-rate command, thrust]
```

## Repair pipeline

### 1. Classify the failure

Determine:

- final valid target gate;
- terminal cause;
- whether timing was healthy;
- whether the localizer was locked, coasting, or relocalizing;
- impact/crossing position in gate coordinates;
- likely minimum intervention horizon;
- whether the failure is physical, perceptual, or mixed.

Only repair episodes whose pre-terminal state history is valid. A vision-loss
failure may still be eligible, but its branch worlds must reproduce the same
belief uncertainty rather than starting from privileged perfect state.

### 2. Create branch snapshots

Construct snapshots at each rollback horizon. A snapshot contains both:

- physical state used by the dynamics model;
- controller belief/observation state used by the policy.

For each nominal snapshot, generate a small initial-state cloud using measured
localizer covariance, model error, timing jitter, and state-estimation bias.
This prevents a repair from depending on one exact decoded state.

### 3. Failure-reproduction gate

Before searching for repairs, replay the original recorded actions through
every relevant model member from each snapshot.

The job is eligible only if the model reproduces the important failure
mechanism often enough to be meaningful, for example:

- same gate/obstacle failure class;
- similar side and height of miss;
- similar time-to-impact;
- similar approach speed and attitude.

Exact centimeter agreement is not required. If the model turns the original
failure into an easy success, its local counterfactuals are not trustworthy;
the episode should instead become real world-model training data.

### 4. Search bounded interventions

Search residual actions around the exact protected controller:

```text
candidate action = protected action + residual correction
```

Do not independently randomize every 30 Hz command. Parameterize smooth
corrections with knots approximately every 0.1-0.2 s and interpolate between
them. Useful search variables include:

- lateral rate correction;
- vertical/pitch correction;
- yaw correction;
- thrust correction;
- command onset and release time;
- optional bounded reference crossing-point offset;
- optional bounded segment-speed scale.

Use CEM or MPPI for the first implementation. Gradient optimization can be
added later, but all final candidates must pass the same ensemble audit.

Search proceeds from low to high intervention authority. Prefer the smallest
action change that produces a robust repair.

### 5. Robust repair score

A repair receives a high score only if it:

- crosses the intended gate in the official direction and sequence;
- maintains clearance under worst-tail perturbations;
- avoids collision and off-course states;
- preserves useful gate visibility/localization confidence;
- produces a controllable position, velocity, heading, and speed for the next
  gate;
- does not materially increase total segment time;
- remains close to demonstrated state/action support;
- works across independently trained world-model versions;
- has low ensemble disagreement.

A conceptual robust objective is:

```text
score = CVaR(
    gate-pass reward
  + future-entry quality
  - segment time
  - clearance risk
  - collision risk
  - localization-drought risk
  - action magnitude and jerk
  - model disagreement
  - support-distance penalty
)
```

CVaR or worst-member scoring is required. Ensemble-mean success is not
sufficient.

### 6. Extend beyond the immediate gate

Every candidate must continue at least 0.5-1.0 s past the gate or until the
next gate becomes well conditioned. Otherwise, the search can "repair" one
gate by creating an impossible next-gate entry.

For chicanes such as gates 9-10, score the pair jointly. For a failure shortly
after a gate event, include the previous crossing and current target in one
repair horizon.

### 7. Acceptance gate

Initial conservative acceptance criteria:

- at least 95% intended-gate success across the complete branch audit;
- no model family below 90% success;
- positive worst-tail clearance chosen from measured live tracking error, not
  nominal geometry alone;
- action residual within the configured trust region;
- support distance below the frozen threshold;
- next-gate entry metrics no worse than successful real examples;
- original failure reproduced by the same model pool used for repair search;
- final result passes a fresh-seed audit not used during search.

These thresholds must be calibrated against live transfer. They may become
stricter if offline repairs remain optimistic.

## Outputs

Each accepted repair is an immutable artifact containing:

- source episode and snapshot identifiers;
- rollback horizon;
- original and repaired action sequences;
- protected-controller actions;
- nominal and perturbed state rollouts;
- per-model pass rate, clearance, time, and next-gate entry metrics;
- ensemble disagreement and support distance;
- random seeds and complete search configuration;
- world-model and dataset hashes;
- repair acceptance/rejection reason.

Synthetic rows must retain an explicit `synthetic_repair=true` marker and may
never be confused with real simulator transitions.

## Use cases

### 1. Behavior-cloning augmentation

Use the first robust action of each repair, or the short repaired sequence, as
a supervised target for states the live policy actually visits. This directly
teaches corrective behavior missing from clean expert laps.

Synthetic repair examples receive lower weight than real successful actions
until live transfer is demonstrated.

### 2. Self-imitation and replay augmentation

Insert robust repaired sub-trajectories into a dedicated demonstration/replay
partition. Prioritize states near recurrent failure gates while retaining real
successful transitions from the rest of the course.

Do not report synthetic gate passes as real completion statistics.

### 3. Counterfactual actor advantage

At a failed real state, compare the recorded action with robust repaired
actions. The estimated difference supplies a high-information actor update:

```text
advantage(repair) = robust repaired return - reproduced failure return
```

Use conservative weighting and clip the resulting actor update inside the
protected residual trust region.

### 4. Critic supervision

Repair branches provide local action ranking and short-horizon return targets
for the critic. They are useful where one-step TD learns slowly because gate
events are many frames apart.

Critic targets must include model uncertainty and must not replace real return
targets wholesale.

### 5. Reference-trajectory repair

Aggregate accepted repairs across repeated failures to estimate a better gate
crossing point, earlier turn onset, or segment-speed profile. A consistent
repair found from many episodes is evidence for modifying the reference line;
one isolated repair is not.

### 6. Controller and authority diagnosis

The minimum successful rollback and residual magnitude identify the likely
problem:

- successful at 0.2 s with tiny residual: trim/margin issue;
- requires 0.7-1.0 s: turn timing or reference geometry issue;
- requires saturated residual: insufficient authority or bad line;
- impossible across all supported actions: model mismatch, localization
  failure, or physically unrecoverable state.

This converts repeated trial-and-error tuning into measurable diagnosis.

### 7. Recovery curriculum

Use repaired branches to train from perturbed states around real failure
approaches. Begin with states close to robust repairs, then expand the initial
state cloud as the policy becomes reliable.

The real simulator still starts at gate 0; curriculum branching remains an
offline training mechanism.

### 8. Active data collection

When ensemble members disagree about which repair works, generate a small set
of safe, macro-level live probes that maximally distinguish the hypotheses.
This directs simulator time toward model identification rather than random
exploration.

### 9. Localization robustness testing

Replay the same physical repair under different belief errors, vision droughts,
and relocalization snaps. This identifies whether the policy succeeds because
of true dynamic robustness or because it assumes unrealistically accurate
localization.

### 10. Near-miss hardening

Run repair search on successful but low-clearance crossings. The objective is
not to turn failure into success, but to trade a small amount of time for a
large increase in worst-tail clearance.

## What synthetic repairs must not be used for

- Do not train the physical world model on its own generated transitions.
  Dynamics models learn from real simulator/VQ1 truth data only.
- Do not count synthetic finishes as evidence of live reliability.
- Do not promote a raw open-loop repair sequence as the flight controller.
- Do not infer official gate events solely from synthetic geometry during live
  operation.
- Do not use repairs from a model that cannot reproduce the source failure.
- Do not let synthetic examples overwhelm real successful demonstrations.

## Actor distillation

The deployed result should be a closed-loop residual actor, not a library of
open-loop motor-command replays.

Train the actor on:

- repaired state/action pairs;
- nearby perturbed branch states;
- original successful expert states;
- original failed states with low or negative weight on failed actions;
- confidence and model-support features where deployable.

After distillation, audit the actor by rerunning all source snapshots and the
entire gates 0-4/full-course suite. The distilled actor must reproduce repair
benefits without regressing previously solved gates.

## Initial proof-of-concept

Start with one repeated, healthy-timing physical failure from gates 0-4.

1. Select a failure with a matching successful episode under the same stack.
2. Export 0.2, 0.4, 0.7, and 1.0 s snapshots.
3. Verify failure reproduction across the current model pool.
4. Run bounded CEM repair search on Gipsydanger.
5. Produce at least one robust accepted repair or a measured reason none
   exists.
6. Distill accepted repairs into a copy of the protected residual actor.
7. Audit all gates 0-4 offline using fresh worlds.
8. If eligible, test champion and candidate interleaved live.

The POC succeeds if the repair-derived actor materially improves the selected
failure class without reducing overall gates 0-4 pass rate or clearance.

## Compute placement

### Gipsydanger

- snapshot branching;
- CEM/MPPI search;
- ensemble and model-version audits;
- repaired demonstration generation;
- actor/critic distillation;
- fresh-seed regression and report generation.

### Local simulator machine

- capture immutable real episodes;
- run paired champion/candidate validation;
- preserve all raw streams on `D:`;
- perform bounded active-identification probes when requested by model
  disagreement;
- avoid heavy training during latency-sensitive live control.

## Risks and mitigations

### Model exploitation

Mitigate with failure reproduction, pooled model versions, CVaR scoring,
support penalties, exaggerated uncertainty, initial-state clouds, and fresh
audit seeds.

### Rollback too late

Search multiple horizons and prefer the earliest low-authority robust repair.

### Repair harms the next gate

Continue the audit beyond the current gate and score next-gate entry state.

### Synthetic-data feedback loop

Never train dynamics on generated repairs. Keep real and synthetic replay
partitions separate and cap synthetic actor/critic weight.

### Actor forgets solved gates

Mix protected successful trajectories into every distillation batch and run a
full earlier-course regression before live testing.

### Localization shortcut

Branch both physical and belief state, include measured vision droughts, and
remove privileged state from deployed actor inputs.

## Implementation tasks

- [x] Define the immutable repair artifact schema.
- [x] Add a failure/snapshot exporter for episode NPZ and step logs.
- [x] Implement original-failure reproduction scoring.
- [x] Implement multi-horizon snapshot clouds.
- [x] Implement smooth residual spline parameterization.
- [x] Implement batched CEM search across pooled model members.
- [x] Add gate, clearance, next-entry, support, and disagreement scoring.
- [x] Add fresh-seed repair audit and acceptance report.
- [x] Add repaired demonstration/replay export with synthetic provenance.
- [x] Add behavior-cloning and critic-weight controls for synthetic repairs.
- [x] Add state-local repair activation with exact protected fallback.
- [x] Add source-episode and gates 0-4 regression audits.
- [x] Run the first gates 0-4 repair POC on Gipsydanger.
- [ ] Accumulate repeated matching real failures before fitting a shared
      structural reference repair.
- [ ] Record an interleaved live champion/candidate result only after a
      candidate passes every offline promotion gate.

## Relationship to the broader campaign

This mechanism is an optional augmentation inside
`VQ2_WORLDMODEL_RL_PLAN.md`. It does not replace the protected champion,
offline-to-live calibration, model acceptance gates, or deterministic live
promotion rules.

Its primary value is converting each expensive real failure into targeted
counterfactual supervision while preserving the successful portions of the
existing controller.

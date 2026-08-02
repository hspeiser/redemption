# VQ2 uncertainty-aware world-model RL plan

Updated 2026-08-01.

## Decision

Proceed with world-model reinforcement learning, but do not treat the learned
model as a perfect replacement simulator. The first deliverable is a locked
gates 0-4 proof-of-concept. Full-course policy training is authorized only
after that controller transfers live with the required speed and reliability.

The approach is a Dyna-style hybrid:

1. learn short-horizon dynamics and localization-error models from real logs;
2. train a bounded residual policy around a proven live controller;
3. optimize robust performance across an ensemble, not one nominal model;
4. test champion and candidate interleaved in the real simulator;
5. append live counterexamples, refit, and repeat for at most two correction
   rounds before making the go/no-go decision.

## Evidence and current baseline

- The deterministic full-course v7/multigate controller has freshly reproduced
  complete laps at 39.264 s and 39.427 s.
- The v13 detector failed 6/6 matched full-course flights and is not the live
  default, despite better offline drought recall.
- The existing gates 0-4 baseline is approximately 9.733 s.
- A previous candidate scored approximately 99% offline but completed only 1/4
  live attempts. This proves that nominal simulated completion is not a valid
  promotion metric by itself.
- Existing world-model error around a 1.07 s horizon is large relative to a
  gate aperture. The model is useful for short-horizon control and data
  generation, but not trustworthy as a single 40-second deterministic rollout.
- `D:\ai-gp\expert_datasets\vq2_13finish_demo_v1.npz` contains 24,950
  transitions and 13 verified completed trajectories, including Gipsydanger
  runs. Slower finishes are valuable support and recovery data; the fastest
  completed lap remains the speed teacher.

## Non-negotiable rules

- Never overwrite a protected champion or its recordings.
- Split datasets by complete episode/session, never by individual frame.
- Keep a frozen validation pool that is never used for model fitting or policy
  selection.
- Separate simulator dynamics error from localization/observation error.
- Keep live flight evaluation-only until a candidate has passed the complete
  offline audit.
- Do not promote on one lucky finish. Promotion requires deterministic
  regression flights.
- Do not introduce a new hand-tuned gate offset after each failed candidate.
  Failed live flights become model data; the same optimization pipeline must
  produce the correction.
- Preserve all camera frames, IMU, localizer states, commands, events, and
  configuration hashes under `D:`.

## System architecture

### 1. Training state

The world model and policy operate on deployable, localizer-derived features:

- local course-frame pose and velocity;
- attitude and body rates;
- next-gate and future-gate geometry in the drone/body frame;
- current gate index and official gate events;
- previous normalized command and actual wire command;
- localizer covariance, landmark age, detector source, and relocalization flag;
- short action and IMU history where required.

Pixels remain in the perception stack and recordings. The first control-world
model does not attempt to predict raw images.

### 2. Action

Use the same canonical action convention as the live harness:

- roll-rate command;
- pitch-rate command;
- yaw-rate command;
- collective thrust.

The stored vector order is `[roll rate, pitch rate, yaw rate, thrust]`.

The learned actor initially outputs a bounded residual over the protected
reference/controller. It does not receive unrestricted full-control authority.

### 3. Dynamics ensemble

Train five to seven independently seeded hybrid members:

- measured analytic rigid-body and rate-loop backbone;
- learned residual for body-frame velocity, attitude, and body-rate changes;
- stochastic localization-corruption model for drift, landmark droughts,
  covariance growth, bias, and relocalization snaps;
- gate-crossing, collision, and off-course outcome heads;
- multi-horizon losses at 1, 4, 8, 16, and 32 control steps.

Ensemble disagreement represents epistemic uncertainty. Policies pay a strong
penalty for entering state/action regions with high disagreement or poor data
support.

## Data program

### Existing data to ingest

- all valid local full-session recordings under `D:\ai-gp\raw_sessions`;
- all episode NPZ and step logs under `D:\ai-gp\training`;
- the 13-finish expert corpus;
- all compatible VQ1 truth runs for motor/rate/drag identification;
- all compatible Gipsydanger VQ1 and VQ2 runs, including failures;
- the latest matched v7 successes, v7 failures, and rejected v13 flights.

### Dataset hygiene

Each transition must carry:

- session, episode, configuration, controller, detector, map, and schedule
  hashes;
- simulator and wall-clock timestamps;
- gate/segment identity;
- terminal type and timing-health status;
- whether it is eligible for dynamics, localization, event-head, or policy
  training.

Exclude reset discontinuities, post-impact falling, invalid sensor intervals,
and simulator-clock stalls from dynamics targets. Retain them as labeled
terminal/infrastructure examples.

### Structured identification flights

When additional real data is needed, use bounded macro perturbations rather
than frame-level random noise:

- small segment speed changes;
- command lead/lag variants;
- small lateral and vertical reference variants;
- bounded thrust/rate changes;
- champion control flights interleaved in the same session.

These produce measurable action consequences while keeping the drone inside a
recoverable region.

## Offline model acceptance gate

Before policy optimization counts, report on frozen sessions:

1. position, velocity, attitude, and body-rate rollout error at 0.13, 0.27,
   0.53, and 1.07 s;
2. gate-crossing, collision, and wrong-side classification quality;
3. reliability calibration: predicted success probability versus realized
   success on held-out trajectories;
4. ensemble calibration: large real errors must coincide with large predicted
   disagreement;
5. intervention ranking: predicted effects of structured probes must have the
   correct direction and useful rank correlation;
6. protected-controller replay must reproduce its time and clearance
   distributions, not merely one nominal trajectory.

Model versions are rejected if prediction error improves while reliability
calibration or intervention ranking regresses.

## Policy-learning stack

### Stage A: imitation initialization

- Behavior-clone the fastest successful trajectory.
- Include all completed Gipsydanger laps as lower-weight support examples.
- Include successful sub-trajectories from failed episodes through
  self-imitation learning.
- Preserve the reference controller as the protected fallback.

### Stage B: robust model-based RL

- Train PPO in the fast ensemble for throughput.
- Optionally train SAC or conservative offline RL on the combined real and
  model-generated replay buffer.
- Optimize worst-member/CVaR return rather than ensemble-mean return.
- Include penalties for ensemble disagreement, out-of-support actions,
  clearance loss, localization drought, and action discontinuity.
- Use official gate order and events as the authoritative success objective.
- Increase residual authority only after reliability is preserved at the
  previous authority level.

### Stage C: live Dyna correction

- Freeze the candidate before live testing.
- Interleave champion and candidate flights in randomized ABBA order.
- Keep every live flight out of training until that A/B round closes.
- Append both successes and failures, refit all ensemble members, re-audit on
  fresh seeds, and regenerate the candidate.
- Permit at most two correction rounds for the gates 0-4 proof.

### Optional augmentation: counterfactual trajectory repair

For healthy-timing live failures, branch the world model from recorded states
0.2-1.0 seconds before impact and search thousands of bounded residual-action
sequences for robust alternatives. Accepted repairs can provide behavior-
cloning targets, self-imitation replay, local critic supervision, reference-
trajectory corrections, recovery curricula, and active-data-collection
requests.

Synthetic repairs remain explicitly labeled, receive less weight than real
transitions, and are never used to train the physical dynamics model. Full
design and acceptance safeguards are in
`VQ2_COUNTERFACTUAL_TRAJECTORY_REPAIR.md`.

## Gates 0-4 proof-of-concept

This is the immediate implementation target. It is deliberately cheaper and
more falsifiable than committing directly to a full-course learned policy.

### Metrics

- Protected baseline: approximately 9.733 s to the official gate-4 crossing.
- Minimum technical proof: at most 8.760 s median, at least 9/10 successful
  locked candidate flights.
- Strong expansion signal: at most 8.273 s median with the same reliability.
- Sub-30-grade signal: at most 7.787 s median.
- No material regression in clearance, timing health, landmark age, or
  localization covariance.

### Protocol

1. Build the offline-to-live calibration table by joining historical live
   sessions with exact configuration/schedule hashes.
2. Re-evaluate every historically flown schedule under the current frozen
   model stack.
3. Optimize bounded residual policies using rotating audit seeds.
4. Run a final never-seen 4,096-world or larger mega-audit.
5. Conduct 4 champion + 4 candidate calibration flights, interleaved.
6. If still eligible, conduct 10 champion + 10 candidate locked flights.
7. Promote only if speed and reliability thresholds both pass.

### POC stop condition

If the candidate wins offline but fails the required live threshold after two
counterexample-driven Dyna rounds, stop full-course expansion. Preserve the
world-model, calibration, and system-identification improvements for planning
and diagnostics.

## Full-course expansion

After the gates 0-4 proof passes:

1. freeze the successful early-course actor;
2. expand the learned residual region segment by segment;
3. use the protected controller to reach the active training segment;
4. prioritize gates 9-10, 11-13, and 15-16 using full-course recordings;
5. require a deterministic full finish before enabling additional RL
   authority;
6. require five deterministic regression laps before promotion;
7. optimize completion time only after reliability is stable.

The initial full-course target is a reliable sub-40 controller. The next gates
are sub-35 and sub-30. A sub-30 candidate may not trade away completion rate to
achieve one lucky lap.

## Compute split

### Gipsydanger

- ingest and normalize the complete historical corpus;
- train all ensemble members with independent seeds;
- run multi-horizon audits and uncertainty calibration;
- generate PPO/SAC experience at high throughput;
- run rotating-seed searches and the final mega-audit;
- retain immutable model, policy, dataset, and report artifacts.

### Local simulator machine

- run champion/candidate live A/B flights;
- preserve full recordings on `D:`;
- monitor timing health and stop on simulator-clock stalls;
- generate only bounded structured-identification probes;
- never train heavy models while latency-sensitive live control is running.

## Implementation task list

- [ ] Freeze and hash the current v7 full-course champion and fresh 39.264 s
      finish.
- [ ] Build one manifest covering local and Gipsydanger recordings.
- [ ] Stamp schedule/config hashes into all future dashboard-history rows.
- [ ] Rebuild the historical offline-to-live calibration table.
- [ ] Define immutable train/validation/test session splits.
- [ ] Train five to seven fresh dynamics/localization ensemble members on
      Gipsydanger.
- [ ] Produce the complete frozen model audit and calibration report.
- [ ] Train a behavior-cloned residual actor from successful trajectories.
- [ ] Train uncertainty-aware PPO candidates for gates 0-4.
- [ ] Run rotating-seed audits and one never-seen mega-audit.
- [ ] Run the 4+4 interleaved live calibration flight set.
- [ ] Perform at most two counterexample-driven Dyna correction rounds.
- [ ] Run the locked 10+10 proof and record the go/no-go decision.
- [ ] If successful, expand the residual policy one course segment at a time.

## Required artifacts

Every model or policy candidate must ship with:

- exact dataset manifest and split hashes;
- model/policy weights and SHA-256 hashes;
- full training configuration and random seeds;
- held-out multi-horizon and reliability-calibration report;
- fresh-seed robust audit report;
- live A/B configuration, episode logs, NPZ files, and raw-session paths;
- explicit promotion or rejection decision with the measured reason.

## Relationship to existing plans

- `VQ2_G0G4_WORLDMODEL_POC.md` remains the detailed POC specification.
- `VQ2_SUB30_CAMPAIGN.md` remains the campaign history and rejected-experiment
  record.
- `VQ2_CHAMPION.md` remains the champion-protection policy, but its artifact
  list should be updated after the fresh 39.264 s finish is formally frozen.

This document defines how those pieces become one uncertainty-aware
world-model RL program.

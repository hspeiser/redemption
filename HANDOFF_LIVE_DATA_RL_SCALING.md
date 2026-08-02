# Handoff: scaling VQ2 RL with every live flight

Updated 2026-08-02.

## Outcome

Build a closed-loop learning system in which every valid live VQ2 flight
improves one or more of:

- the dynamics and localization world-model ensemble;
- the residual actor and critic;
- the reference trajectory and segment-speed schedule;
- failure-specific recovery supervision;
- offline-to-live reliability calibration.

The current controller can finish the course. The next objective is not to
replace it with unconstrained online RL. The sustainable path is to preserve
that controller as a protected champion, learn conservative residual
improvements offline, screen them at scale, and promote only improvements that
repeat in the live simulator.

```text
live champion/candidate flights
        |
        v
immutable synchronized episode corpus
        |
        +--> supervised world-model/localization training
        +--> behavior cloning and self-imitation
        +--> offline actor/critic updates
        +--> counterfactual repair jobs
        |
        v
short-horizon ensemble simulation and fresh-seed audits
        |
        v
frozen challengers
        |
        v
interleaved live champion/challenger tests
        |
        +--> promote repeatable winner
        +--> ingest all successes and failures, then repeat
```

## Important distinction

The world model is an **offline training and candidate-screening tool**. It is
not currently part of live inference. Live flight uses the camera/IMU
localizer, reference controller, and trained residual actors.

Do not ask one learned model to predict a complete lap. Existing error near a
one-second horizon is large relative to gate clearance. Use independently
trained ensembles for short rollouts, uncertainty estimates, action ranking,
and local counterfactuals. Long-horizon policy quality must come from
closed-loop rollouts, repeated replanning, real demonstrations, and live
validation.

## Current artifacts to preserve

Before training anything new, freeze and hash the exact currently accepted
full-course configuration and recordings. Relevant locations include:

- live recordings: `D:\ai-gp\raw_sessions`;
- live episode/training logs: `D:\ai-gp\training`;
- completed-run corpus:
  `D:\ai-gp\expert_datasets\vq2_13finish_demo_v1.npz`;
- current launcher: `.remote/launch_live_full17_fastprefix_abba.ps1`;
- primary early-course PPO residual:
  `worldmodel/ppo_multimodel_segmentcredit_v8/best.pt`;
- specialized late-course PPO residual:
  `worldmodel/ppo_all17_multimodel_v3_late_safe/best.pt`;
- broader design: `VQ2_WORLDMODEL_RL_PLAN.md`;
- local failure repair: `VQ2_COUNTERFACTUAL_TRAJECTORY_REPAIR.md`;
- protected-artifact rules: `VQ2_CHAMPION.md`.

Do not train into, rewrite, or silently replace protected artifacts.

## 1. Canonical episode dataset

### Required data per control step

Every future episode should retain:

- session, episode, controller, actor, critic, model, detector, map,
  trajectory, schedule, and configuration hashes;
- simulator timestamp, wall timestamp, camera timestamp, and action-send time;
- localizer pose, velocity, attitude, covariance, landmark age, measurement
  source, relocalization state, and detector confidence;
- raw IMU history and previous canonical actions;
- actual command sent on the wire, including clipping and actuator scaling;
- reference pose, velocity, crossing point, target action, cursor, and tracking
  error;
- current gate, future-gate geometry, official gate events, and course progress;
- reward components, terminal flag, terminal class, crash/wrong-way/reset event;
- camera-frame path and visual detections;
- control-loop delay, camera age, dropped packets, simulator-clock stalls, and
  other session-health signals.

The action convention must remain:

```text
[roll-rate command, pitch-rate command, yaw-rate command, thrust]
```

### Episode-level metadata

Each episode receives:

- official finish/pass outcome and time;
- gates reached and per-gate split times;
- crossing position, velocity, attitude, and clearance for every gate;
- failure gate, side/height of miss, and failure classification;
- timing-health classification;
- whether it is eligible for dynamics, localization, actor, critic, event, or
  infrastructure training;
- immutable source paths and file hashes.

### Preserve sequences

Do not reduce the corpus to shuffled independent rows. VQ2 is partially
observable. Store sequence windows long enough to cover IMU propagation,
localization droughts, action latency, and turn initiation. Keep episode and
session boundaries intact in every data split.

### Data eligibility

Use healthy pre-impact flight for physical dynamics. Exclude reset
discontinuities, post-impact falling, simulator pauses, stale UDP intervals,
and invalid timestamps from physical-dynamics targets. Preserve them with
explicit labels for terminal detection, runtime robustness, and infrastructure
diagnosis.

Failures are valuable, but they are not all the same:

- physical misses train dynamics, critic, and failure/event heads;
- healthy near misses train clearance and risk estimates;
- localization droughts train observation-error and robustness models;
- simulator/compute failures train health monitoring, not aerodynamics;
- successful segments train actor imitation and prevent world-model
  pessimism;
- faster-than-champion segments receive high self-imitation priority.

## 2. Dataset manifest and immutable splits

Create one manifest that indexes local and Gipsydanger recordings without
copying or altering protected source files. Split by complete session:

- training sessions;
- model-validation sessions;
- policy-selection sessions;
- frozen final test sessions.

Never split frames from one flight across train and validation. Never select a
policy against the frozen final test set.

Every generated dataset must record:

- source manifest hash;
- filtering rules and code revision;
- train/validation/test session IDs;
- transition and episode counts by gate and terminal class;
- normalization statistics derived from training data only.

## 3. World-model ensemble

### Model state

Use deployable localizer-derived features rather than privileged position:

- local course-frame belief and uncertainty;
- body-frame velocity, attitude, and rates;
- gate-relative next/future geometry;
- recent IMU, actions, timing, and landmark history;
- reference tracking state and current gate progress.

For training diagnostics, retain true/privileged state where available, but do
not make the deployed actor depend on it.

### Model outputs

Predict distributions over:

- state deltas at 1, 4, 8, 16, and 32 control steps;
- localizer-belief drift and covariance growth;
- gate crossing, collision, off-course, and observation-drought events;
- next-gate entry state and clearance risk;
- action latency or effective-command uncertainty.

Train five to seven independently seeded members, ideally across more than one
model family. Ensemble disagreement is an epistemic-risk signal, not merely a
debug metric.

### Safe use

- Prefer rollouts of approximately 0.25-0.75 seconds.
- Permit longer rollouts only where held-out calibration supports them.
- Penalize ensemble disagreement and distance from demonstrated state/action
  support.
- Optimize CVaR or worst-member performance, not nominal mean return.
- Never train the physical dynamics model on its own synthetic transitions.
- Retrain on both live successes and failures so the model does not become
  locally pessimistic around recurrent failures.

### Acceptance report

For every version, report on frozen sessions:

- position, velocity, attitude, and rate error by horizon;
- crossing/collision-event precision, recall, and calibration;
- predicted versus realized success probability;
- whether high real error coincides with high ensemble disagreement;
- intervention rank correlation for known live schedule/action changes;
- errors broken down by gate, speed, detector state, and timing health.

Reject a model that improves average position error but worsens reliability
calibration or intervention ranking.

## 4. Policy architecture

Keep the protected controller and reference action. Train a recurrent residual
actor:

```text
live action = protected reference/controller action
            + gate-conditioned bounded residual
```

Actor observations should include:

- gate-relative pose/velocity and localizer uncertainty;
- current and future gate geometry;
- previous actions and short IMU/action history;
- reference action and tracking error;
- current gate plus continuous descriptors such as next-turn angle, climb,
  distance, curvature, and expected segment speed.

Keep gate identity initially because the course has distinct hazards, but add
continuous geometry descriptors so data can transfer across gates and future
maps.

The actor should be recurrent or receive explicit history. A memoryless actor
trained on one-frame beliefs will learn around localizer noise rather than the
underlying trajectory.

## 5. Offline learning sequence

### Stage A: behavior cloning

- Clone the fastest reliable live run at the highest weight.
- Include every completed run as lower-weight support data.
- Include every clean successful sub-trajectory from failed runs.
- Weight faster-than-champion gate segments more strongly.
- Mix protected champion data into every batch to prevent forgetting.
- Validate exact observation/action reconstruction before optimization.

This stage should reproduce the champion deterministically before any RL
objective is introduced.

### Stage B: conservative offline RL

Use IQL or AWR as the initial real-data policy-improvement method. They are a
better first fit than unrestricted offline SAC because the dataset is narrow
and mostly near existing trajectories.

Recommended ingredients:

- 8-16-step returns or Retrace for gate-scale credit propagation;
- fixed Markov terminal penalties rather than return-dependent terminal
  ceilings;
- prioritized replay by gate and outcome;
- self-imitation of successful live sub-trajectories;
- advantage-weighted actor regression with clipped weights;
- critic ensembles and conservative/out-of-support penalties;
- separate real-demo, real-live, and synthetic-repair partitions;
- balanced sampling so gates 0-4 do not overwhelm gates 11-16.

The critic can learn from all valid real transitions. The actor should learn
primarily from demonstrated or positive-advantage actions until the model and
critic are calibrated.

### Stage C: uncertainty-aware model-based fine-tuning

Use the world-model ensemble for short imagined continuations from real states:

- start rollouts from real replay states;
- terminate imagined rollout on high disagreement or support distance;
- randomize latency, localization errors, model member, drag, thrust response,
  and initial state within measured distributions;
- optimize tail reliability before average speed;
- keep residual authority bounded;
- periodically anchor the actor to real successful behavior.

Conservative SAC can be used here after BC/IQL initialization. Do not begin
from random actions, and do not allow synthetic samples to dominate real
replay.

## 6. Segment curriculum

Treat a lap as 17 reusable segment problems rather than one sparse 37-second
episode.

For each gate segment, measure:

- entry-state distribution;
- time from previous gate;
- crossing position and clearance;
- exit-state quality for the next gate;
- failure probability and mechanism;
- actor/reference disagreement;
- model uncertainty and live/offline calibration.

Train only one or two active segments at a time while protected behavior
controls the rest. A candidate segment must preserve the next gate's entry
state. Saving 0.2 seconds on a straight is not an improvement if it makes the
following turn marginal.

Suggested progression:

1. repeatably fast straight segments with large measured clearance;
2. existing marginal but solved turns where the actor already has evidence;
3. recurrent failure classes with matching successes;
4. chicanes or late gates requiring joint two-gate optimization;
5. whole-course fine-tuning only after segment policies compose reliably.

Counterfactual trajectory repair remains appropriate for repeated,
healthy-timing failures. It is local supervision, not an independent promotion
path.

## 7. Reward and critic targets

Use official gate events as authoritative. The objective is lexicographic:

1. correct ordered completion;
2. completion reliability and clearance;
3. lower total time and faster splits.

Useful components include:

- course-progress delta;
- gate and finish bonuses;
- per-step time cost;
- fixed crash/wrong-way/off-course terminal penalties;
- clearance and controllable-next-entry penalties;
- backtracking and stagnation penalties;
- localization-risk and model-disagreement penalties for imagined data;
- small action jerk/support penalties.

Do not let dense noisy EKF progress dominate authoritative gate events. Log
every reward component separately so critic failure can be diagnosed.

## 8. Candidate generation and offline audits

Each training cycle may generate many checkpoints, but only a small number
become challengers. Screen them using:

- deterministic replay/regression on all earlier solved gates;
- fresh randomized ensemble worlds;
- rotating audit seeds between training stages;
- one final never-seen mega-audit;
- worst-model/CVaR completion and clearance;
- segment time and full-lap time;
- support distance and action saturation;
- localization drought and timing perturbations.

Do not repeatedly select against one fixed audit pool. Preserve every audit
configuration and seed.

## 9. Live champion/challenger protocol

Never evaluate a challenger alone. Interleave within the same healthy session:

```text
champion -> challenger -> challenger -> champion
```

or randomized equivalent. Champion controls reveal session degradation,
simulator hitching, or vision-health changes.

Promotion requires:

- deterministic full-course completion;
- no material completion-rate regression;
- no new gate-specific failure concentration;
- repeated speed improvement, not one lucky lap;
- comparable timing/localization health;
- preserved checkpoint, config, hashes, episodes, and raw recording.

Use a small calibration round first, then a larger locked regression round.
Keep the previous champion available for immediate rollback.

After the A/B round closes, ingest **all** champion and challenger episodes.
Challenger successes teach improvements; challenger failures define the next
model/critic boundary; champion controls improve calibration.

## 10. Continuous training cycle

A practical recurring cycle is:

1. Fly 10-30 controlled live episodes when the simulator is healthy.
2. Close the session and build/validate its immutable manifest entries.
3. Retrain or incrementally update world-model members on Gipsydanger.
4. Run BC/IQL/critic updates with 4-8 or more offline updates per newly
   ingested real transition, subject to validation.
5. Generate short ensemble rollouts from high-value real states.
6. Train several gate-scoped residual challengers.
7. Audit on rotating and never-seen randomized worlds.
8. Live-test only the strongest one or two challengers, interleaved with the
   champion.
9. Promote a repeatable winner or record a measured rejection.
10. Append all new evidence and repeat.

Heavy model and policy training belongs on Gipsydanger. The local simulator
machine should run latency-sensitive inference, recording, and controlled live
tests without competing training workloads.

## 11. Offline-to-live calibration

This is required for meaningful scaling. Every dashboard/history row must
carry the exact schedule/config/model hash. Retroactively join historical live
sessions to their saved configurations where possible, re-evaluate each flown
policy under the current frozen offline stack, and fit calibration curves:

- offline completion probability versus live completion rate;
- offline segment failure probability versus live gate failure rate;
- offline predicted time/clearance versus live time/clearance;
- model disagreement versus live error.

This converts arbitrary offline thresholds into measured promotion thresholds
and lets the entire historical corpus contribute to candidate selection.

## 12. Monitoring and artifacts

The lightweight dashboard should show:

- current champion and challenger hashes;
- live run number, outcome, time, gate reached, and terminal reason;
- best accepted time and completion rate over time;
- per-gate pass rate and split distributions;
- policy-learning updates and checkpoint creation;
- world-model version and held-out metrics;
- candidate offline audit versus live A/B result;
- simulator, camera, inference, and localization health.

Every promoted/rejected candidate must ship with:

- dataset and split manifest hashes;
- exact training config and code revision;
- actor, critic, and model hashes;
- offline audit and calibration reports;
- live A/B session paths and recordings;
- explicit promotion/rejection reason.

## 13. Main-agent implementation order

Execute in this order:

- [ ] Freeze/hash the current accepted full-course policy, launcher, runtime
      assets, finish episode, and raw recording.
- [ ] Add exact config/schedule/model hashes to all future episode and dashboard
      records.
- [ ] Build the read-only corpus manifest across local `D:` and Gipsydanger.
- [ ] Implement the canonical sequence-dataset exporter and eligibility flags.
- [ ] Define immutable session-level train/validation/policy-test/final-test
      splits.
- [ ] Produce corpus counts by gate, outcome, controller, detector, and health.
- [ ] Validate observation/action reconstruction against existing live runs.
- [ ] Train a recurrent BC residual actor that exactly reproduces the protected
      champion before RL.
- [ ] Add successful live sub-trajectories to a prioritized self-imitation
      partition.
- [ ] Implement fixed terminal targets and 8-16-step/Retrace critic targets.
- [ ] Train an IQL/AWR residual candidate on real data only.
- [ ] Train/retrain five to seven short-horizon ensemble members on
      Gipsydanger.
- [ ] Produce the frozen multi-horizon, event, calibration, and intervention
      audit.
- [ ] Add uncertainty-limited imagined continuations and conservative SAC
      fine-tuning.
- [ ] Select one gate/segment with repeated evidence and train a scoped
      challenger.
- [ ] Run rotating-seed audits plus a never-seen mega-audit.
- [ ] Run an interleaved live champion/challenger calibration set.
- [ ] Ingest all A/B results, promote only a repeated winner, and begin the next
      cycle.

## First concrete milestone

The first milestone is not a new full-lap record. It is proof that the loop
learns from live evidence:

1. select one segment with multiple healthy successes and failures;
2. train a real-data-only recurrent BC+IQL residual challenger;
3. demonstrate a meaningful offline segment improvement on frozen sessions;
4. pass fresh ensemble audits without earlier/later segment regression;
5. beat the protected segment live in an interleaved test;
6. ingest that test and reproduce or improve the result in the next cycle.

Once this succeeds, scaling to other gates is largely data and compute. Until
it succeeds, adding more live episodes without ingestion, versioning, and
offline retraining is collection—not learning.

## Definition of success

The system is genuinely learning when:

- every live session appears automatically in a versioned training manifest;
- new model versions become better calibrated on untouched real sessions;
- actor updates improve held-out live segments, not only synthetic rollouts;
- champion/challenger outcomes match offline ranking often enough to guide
  selection;
- repeated live failures produce targeted policy/model changes rather than
  manual gate offsets;
- accepted lap time falls while deterministic completion remains protected.

This document is the execution overlay for `VQ2_WORLDMODEL_RL_PLAN.md`. The
existing champion, counterfactual-repair, vision, and line-optimization plans
remain authoritative for their respective subsystems.

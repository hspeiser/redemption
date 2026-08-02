# VQ2 live-data flywheel status

Updated 2026-08-02 after cycle `live36-v30-g5g6`.

## Protected outcome

- A new accepted official 36.826946258 s outcome is frozen at
  `D:\ai-gp\champions\vq2_36s_20260802`.
- The complete artifact manifest covers 9,276 files and has SHA-256
  `40232920af6b8273998c6da56206c6de7b723fc87e9a05582bc79aac1c8d6bae`.
- No learned actor or counterfactual repair from this cycle was flown or
  promoted.
- The simulator machine was kept free of model-training load.

## Latest live collection

The fixed 36-second behavior was repeated eight times under one controller
identity. All eight episodes were timing-healthy: one official finish at
36.826946258 s, four gate-5 failures, one gate-6 failure, and two gate-3
failures. This produced matched successes and failures without injected noise
or hand labels.

A diagnostic +0.10 m gate-5 / -0.10 m gate-6 reference offset was then tested.
It completed 2 of 7 timing-healthy candidate flights at 36.963943481 s and
36.888019561 s. One additional episode was aggregate timing-unhealthy and is
quarantined. The offset is retained for a larger interleaved regression but is
not promoted. The first interleaved controls used the historical reference-only
champion behavior because the protected arm suppressed the PPO residuals. That
made them useful session-health controls, but not valid comparisons against the
new hybrid champion. The launcher and trainer now preserve residual actors for
hybrid protected champions.

## Data and split layer

- Canonical corpus manifest: 10,110 indexed artifacts.
- Immutable split registry SHA-256:
  `b6f835c5085d29125f50468d8ab1dd2dedd843b8bf19273531a03ae562765336`.
- The world-model builder now consumes that registry. Known final-test
  sessions are excluded from routine training and selection; newly collected
  sessions are train-only until a new registry generation is explicitly
  created.
- Current-era v30 training corpus: 207,087 timing-healthy real transitions.
- Broader untouched audit corpus: 44,693 validation and 30,823
  policy-selection transitions, covering gates 0-16.
- The builder now rejects an explicit episode-level timing quarantine before
  reading transitions. This prevents simulator-step p95/max failures from
  leaking in when individual packet-age flags happen to remain true.

## World model v30

Artifact SHA-256:
`fe21bb15aafd8d31c99a258a851c26bd3ae7d2a12b8eb6c040e6f15e868abf0a`.

On 26,887 identical untouched 32-step starts:

| Metric | v25 | v28 | v30 | Decision |
|---|---:|---:|---:|---|
| Position median | 0.348 m | 0.317 m | 0.317 m | v30 best |
| Position p90 | 0.781 m | 0.743 m | 0.740 m | v30 best |
| Velocity median | 0.339 m/s | 0.307 m/s | 0.306 m/s | v30 best |
| Velocity p90 | 0.808 m/s | 0.704 m/s | 0.699 m/s | v30 best |
| Attitude median | 0.538 deg | 0.607 deg | 0.579 deg | v25 best |
| Attitude p90 | 1.788 deg | 1.857 deg | 1.810 deg | v25 best |

Decision: use v25, v28, and v30 as independent pooled audit families. v30 is
the strongest translational model, while v25 preserves a useful
attitude-strong disagreement family. None is used for live inference.

## Real-data policy attempts

The gate-5 and gate-10 recurrent BC+IQL/AWR actors were rejected before
simulation. In both cases the untouched residual actor had lower held-out
action error than the learned actor. This means the currently logged residual
commands are not a stable imitation target across controller configurations.
The critic fits and failure rows remain useful; the actors are quarantined.

Late-gate inspection found a configuration-distribution mismatch: several
validation successes use the protected zero-residual controller while faster
policy-selection successes use a nonzero specialist. Future BC validation
must be controller/config-hash matched.

## Previous world model v28 cycle

Artifact SHA-256:
`7df27fc6d052761ee00dd5f8b1fd502f5d4a9ae1163c65827ff93e12865fc001`.

On 27,327 identical untouched 32-step starts versus v25:

| Metric | v25 | v28 | Result |
|---|---:|---:|---|
| Position median | 0.347 m | 0.317 m | 8.8% better |
| Position p90 | 0.780 m | 0.741 m | 5.0% better |
| Velocity median | 0.338 m/s | 0.307 m/s | 9.2% better |
| Velocity p90 | 0.806 m/s | 0.702 m/s | 12.9% better |
| Attitude median | 0.537 deg | 0.605 deg | worse |
| Attitude p90 | 1.783 deg | 1.852 deg | worse |

Decision: retain v28 as a complementary translational model in the pooled
offline ensemble. Do not replace v25 wholesale and do not use the learned
model for live inference.

## Gate-5 / gate-6 counterfactual cycle

The repair pipeline contained two gates-0-4 assumptions that made every gate
5+ branch terminate immediately: a hard-coded five-gate race horizon and a
five-entry teacher configuration. Both are fixed and regression-tested for
all gates through 16.

- Gate 5: v25 reproduced the failure only at the 1.5 s rollback; v28 did not.
  It is a model-boundary example, not eligible for synthetic repair.
- Gate 6: v25 and v28 reproduced the real failure at all tested rollback
  horizons (0.2, 0.4, 0.7, 1.0, and 1.5 s).
- Action-only gate-6 repair fresh audit: 62.7% worst-family success, rejected.
- Action+geometry gate-6 repair fresh audit: 65.2% worst-family success,
  0.196 m minimum p10 clearance, rejected against the 95% floor.

The geometry search preferred approximately +0.106 m lateral, -0.041 m
vertical, and 0.984 speed scale, but this is diagnosis only. It must not be
copied into the live configuration from one failed episode.

## Legacy Gipsydanger finishes

Twelve archived official full-course control proofs were located under
`C:\Users\henry\codex_export_vq2_finishes_20260731`. They contain 5 Hz replay
belief/action rows plus a separate raw-IMU stream. They are preserved as a
legacy-real partition. They must be timestamp-synchronized and resampled
before entering the current 30 Hz dynamics or actor datasets.

## Next closed-loop cycle

1. Run a larger interleaved baseline/offset regression after the next healthy
   simulator relaunch; the current 2/7 diagnostic rate is not promotion proof.
2. Ingest all outcomes; failures update dynamics and critic targets while
   successes pin the model and actor against pessimism.
3. Add a stable behavior-profile identity that excludes run length/output
   paths so controller-matched policy validation spans multiple sessions.
4. Build behavior-matched BC validation pools. Do not optimize residual
   action MAE across incompatible protected/specialist configurations.
5. Re-run gate-scoped IQL or counterfactual repair only where repeated real
   successes and failures provide a calibrated target.
6. Require fresh pooled audit success before creating an interleaved live
   challenger. Until then, keep the 36.826946258 s stack protected.

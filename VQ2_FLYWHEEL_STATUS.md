# VQ2 live-data flywheel status

Updated 2026-08-02 after cycle `live37-v28-g6repair`.

## Protected outcome

- The accepted 37.00 s configuration and recording remain frozen.
- No learned actor or counterfactual repair from this cycle was flown or
  promoted.
- The simulator machine was kept free of model-training load.

## Data and split layer

- Canonical corpus manifest: 9,961 indexed artifacts.
- Immutable split registry SHA-256:
  `60465cebaa8bb114cef17f0ec02032e9b83aedd18b764d49ac3cbe19e898be5c`.
- The world-model builder now consumes that registry. Known final-test
  sessions are excluded from routine training and selection; newly collected
  sessions are train-only until a new registry generation is explicitly
  created.
- Current-era v28 training corpus: 201,755 real transitions.
- Broader untouched audit corpus: 44,937 validation and 31,520
  policy-selection transitions, covering gates 0-16.

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

## World model v28

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

1. Collect a fresh healthy champion-control batch with full recording after a
   simulator relaunch, emphasizing repeated gate-5/gate-6 outcomes under one
   exact config hash.
2. Ingest all outcomes into the immutable corpus; failures update dynamics and
   critic targets, successes pin the model and actor against pessimism.
3. Train v29 with explicit attitude/event calibration and retain v25/v28 as
   independent audit families.
4. Build controller-hash-matched BC validation pools. Do not optimize residual
   action MAE across incompatible protected/specialist configurations.
5. Re-run gate-scoped IQL or counterfactual repair only where repeated real
   successes and failures provide a calibrated target.
6. Require fresh pooled audit success before creating an interleaved live
   challenger. Until then, keep the 37.00 s stack protected.


# VQ2 deployment-controller parity oracle

## Purpose

This package is the independent acceptance oracle for the batched deployment
controller being implemented for suffix optimization.  Candidate search must
not begin until the port passes both golden-action and paired-rollout parity.

## Golden fixture

`tests/fixtures/vq2_35p37_teacher_parity_v1.npz` contains all 1,067 control
steps from the official 35.374649-second episode.  It includes:

- 53-dimensional deployable observations
- teacher action, routed actor output, and final normalized command
- integer and fractional selected-reference rows
- reference-segment bounds
- target gate and actor source
- per-gate velocity, thrust, lateral-gain, blend, and residual settings
- raw/clipped lateral and longitudinal feedback
- trajectory-blend and residual-routing state

The adjacent JSON manifest records the source/config hashes and fixture hash.
Regenerate it with `scripts/build_vq2_teacher_parity_fixture.py` only when the
frozen champion itself intentionally changes.

Field semantics are intentionally distinct: `reference_row_int` is the true
nearest-reference cursor used for row parity. `selected_reference_row` is the
weighted fractional **action** row after applying action lead; it must not be
used as the cursor. `selected_reference_row_int` is the corresponding archived
integer action-row diagnostic.

The parity harness verifies the candidate config's SHA-256 against the
fixture manifest before replay. A controller replay under a merely similar
config is not parity evidence; in particular, per-gate offsets affect adjacent
segments through interpolation.

Golden-action acceptance requires:

1. Exact actor source, target gate, reference row, segment bounds, and routing
   booleans.
2. Maximum absolute final-action error no greater than `1e-5`.
3. Component-level comparison of teacher action, actor output, gain gathers,
   feedback terms, blend, and residual correction so the first divergence is
   attributable.

## Same-state shadow probe

Golden-action replay verifies controller composition on recorded inputs, but
it cannot prove that the vectorized evaluator constructs those inputs with the
same semantics during a rollout. Before paired outcome audits, run
`scripts/liveteacher_shadow_probe.py` for at least 64 worlds. The trusted scalar
learner drives each world while the batched port computes a shadow action from
the identical position belief, velocity, attitude, previous action, and target.

Acceptance requires zero worlds with an action divergence above `1e-4` and a
p95 per-world maximum action error no greater than `1e-5`. Both controllers,
the plant, and the residual ensemble must run on the same device. A CPU and a
CUDA rollout seeded with the same integer do not share an equivalent random
tape and are not paired evidence.

This layer is permanent. It caught a deployment-only position reconstruction
rule that golden replay cancelled by construction: the localizer's gate vector
uses runtime-map gate centers, while the learner rebuilds position using demo
gate centers. The batched controller must therefore reproduce
`p + (demo_gate - map_gate)` before selecting and tracking its reference.

## Paired rollout comparator

`scripts/compare_vq2_paired_audits.py` consumes baseline and port per-world
NPZ files with:

- `world_id`
- `finished`
- `finish_time_s`
- `failure_gate`

It requires identical world ordering and reports:

- paired finish-rate difference with bootstrap 95% interval
- baseline-only and port-only finishes
- paired finish-time difference among jointly finished worlds
- Jensen-Shannon distance between terminal/failure-gate histograms

Use 256 paired worlds for development and at least 768 fresh paired worlds for
the final parity gate.  The final target is finish-rate difference within
approximately 3 percentage points, median paired time within 0.15 seconds,
and a failure histogram that agrees quantitatively rather than merely sharing
the same total finish rate.

`FastVQ2Env` now passes `prev_action` and `target` to a backbone that declares
`needs_extras = True`; both are required by the deployed teacher. The PPO
evaluator can write the required per-world artifact with
`--eval-world-dump PATH`. Its schema is:

- `world_id`: stable audit-seed-derived identity
- `finished`: Boolean terminal result
- `finish_time_s`: elapsed time, NaN for failures
- `failure_gate`: terminal target gate, -1 for finishes
- `min_clearance_m`: minimum observed gate clearance

For the new port arm, add `--batched-live-teacher-composition`. This is an
eval-only mode: the batched teacher owns the complete primary/secondary actor
routing and the evaluator sets its outer residual scale to zero, preventing a
double-applied PPO correction.

Do not use the legacy `aigp.fastsim.live_teacher.LiveTeacherController` output
as the golden controller merely because it is vectorized; Layer 1 already
showed that implementation is not action-exact. The baseline Layer-2 artifact
must come from the trusted scalar/deployed evaluator selected by the parity
work, under the identical world registry and evaluator settings.

After producing both NPZ files, enforce the staged gate with:

```powershell
.\.venv-train\Scripts\python.exe scripts\accept_vq2_layer2_parity.py `
  --layer1 data\lineopt\liveteacher_parity_layer1.json `
  --shadow data\lineopt\liveteacher_shadow_probe.json `
  --baseline path\to\baseline_worlds.npz `
  --candidate path\to\port_worlds.npz `
  --stage development `
  --out data\lineopt\liveteacher_parity_layer2_development.json
```

Change `--stage` to `final` only for the fresh 768+-world audit. The acceptance
report hashes Layer 1 and both world artifacts, records the Git revision, and
exits non-zero unless all three statistical criteria pass. Candidate search
remains blocked until that report says `PASS: true`.

## Resolved parity failures

The certification campaign closed three independent defects:

- A gate-5/6 action residue came from evaluating the wrong configuration;
  SHA-256 config gating exposed the missing per-gate lateral offsets.
- The first-step attitude discrepancy came from a float32 SO(3) logarithm at
  tiny angles. Using the analytic half-angle limit reduced Layer-1 maximum
  action error to `3.64e-6` across all 1,067 recorded steps.
- The same-state shadow probe exposed the runtime-map versus demo-gate
  position reconstruction described above. After reproducing that behavior,
  the 256-world development audit matched exactly: the same 17 finishes,
  identical `35.933 s` median, zero discordant worlds, and terminal-histogram
  Jensen-Shannon distance `0.0`.

These results certify controller parity. They do not claim that the learned
world ensemble is perfectly calibrated to the live simulator.

Final certification on the untouched 768-world seed also passed exactly:
both implementations finished the same 46 worlds (`5.9896%`), shared a
`35.400 s` median and `36.533 s` p90, had zero discordant outcomes, and
produced identical terminal-gate histograms (`JS = 0.0`). The permanent
64-world shadow gate also passed with zero divergences above `1e-4` and a
`1.55e-6` p95 per-world maximum action error. Deployment-faithful candidate
search is therefore unblocked; the frozen 35.37-second champion remains the
promotion baseline.

## Gate-10 handoff pool

`data/lineopt/handoff_pool_13finish.json` is suitable for the suffix search:

- 13 unique completed-run sources
- speed range 3.07–5.16 m/s
- position standard deviation `[0.58, 0.43, 0.24]` m
- velocity standard deviation `[0.50, 0.41, 0.23]` m/s
- full-rank joint position/velocity covariance
- median attitude deviation 2.68 degrees; maximum 15.28 degrees

Bootstrap the empirical rows and apply small measured start jitter (the
existing 0.08 m position and 0.20 m/s velocity noise is appropriate).  Do not
replace the correlated empirical states with independent axis-wise Gaussian
sampling.

## Ownership boundary

The oracle work does not modify `train_vq2_sac_live.py`,
`aigp/fastsim/live_teacher.py`, or the suffix optimizer's batched controller.
Those remain available for the controller-port implementation without merge
overlap.

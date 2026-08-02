# VQ2 sub-30 campaign state

Updated 2026-07-31.

## Protected baseline

- Artifact: `D:\ai-gp\champions\vq2_40s_20260731`
- Proven official finish: 39.98965 s
- `latest.pt` SHA-256:
  `9710F18A19FDC41C9537305EA3D442A4A053312A9246AF69EB962BE32506E59F`
- Never overwrite this directory. A candidate is promoted only after it
  finishes and passes deterministic regression laps.

## Expert data

- Unified 13-finish dataset:
  `D:\ai-gp\expert_datasets\vq2_13finish_demo_v1.npz`
- 24,950 transitions: the local 39.99 s lap plus 12 verified completed
  Gipsydanger laps.
- The slower Gipsydanger laps provide recovery and state-coverage support;
  they are not the speed reference.

## Findings that changed the campaign

1. `gatenet_v13drought_best.pt` is not a universal replacement for v7.
   Recorded replay showed worse gate 3-5 fusion coverage, so the protected
   stack stays on `gatenet_v7_best.pt`.
2. A 20k-update offline AWR candidate was rejected before live flight. Its
   p95 normalized residual was only about 0.002, so it could not materially
   change the flown command.
3. The delivered 0.25 m-clearance cap-8 line is not live-safe for this stack.
   Four deterministic drop-in attempts passed gates 0-1 and died at gate 2;
   later repeats clipped gate 1. Vision and timing were healthy. The planned
   line targets roughly 0.5 m off-center at gate 1, leaving less margin than
   the measured 0.2-0.3 m live tracking spread.
4. Increasing gate-1 lateral gain made the line worse by turning into the
   gate-1 panel before the official crossing. This is a reference-margin
   problem, not missing controller authority.
5. The old fastsim residual evaluation backbone double-converted canonical
   actions as MAVLink commands and expected obsolete `pos`/`vel` keys. That
   screen was invalid. The action convention and demo conversion are fixed
   locally, but fastsim still does not reproduce the live champion closely
   enough to authorize promotion without real deterministic flights.

## Clearance-constrained line experiments

The first replacement used:

- speed cap: 10 m/s
- planned clearance: 0.50 m
- corrected gate-9/gate-15 map
- measured dynamics, obstacle cylinders, estimator noise, relocations, and
  domain randomization
- output prefix: `D:\ai-gp\lineopt\safe_cap10_c050`

Its final 512-world surrogate validation reached 98.8% completion, but live
flight still exposed an over-optimistic rate model. Seven new full-record
sessions measured the real command-to-gyro loop from 18,046 aligned
transitions:

- old rate gains: `[-2.78, -2.73, -2.50]`
- measured gains: `[-2.51, -2.62, -2.00]`
- measured time constants: `[0.0280, 0.0221, 0.0628]` seconds

The old surrogate overestimated yaw authority by about 25%. The corrected
model is `data/fastsim_model_v3_live.json`, reproducibly fitted by
`scripts/fastsim_raw_rate_sysid.py`.

The v3 re-optimized line is:

- raw optimizer artifact:
  `D:\ai-gp\lineopt\safe_cap10_c050_v3_best.npz`
- live demo artifact: `data/vq2_lineopt_demo_safe_cap10_c050_v3.npz`
- 512-world surrogate result: 100% completion, 41.60 s median, 37.60 s best
- clean surrogate result: 40.97 s
- live result with no new per-gate tuning: gate 1 passed 3/3, gate 2 reached
  3/3, and gate 4 reached 1/3

That is a real transfer improvement over the v2 line, which repeatedly died
at gate 1. It is not promotable: the replay-style controller still misses
gate 2 laterally and is too sensitive to nearest-row action phase.

Rejected local patches:

- gate-2 lateral gain 2.0: feedback saturated but did not reliably clear
- gate-2 action lead -8 rows: caused immediate inversion after gate 1
- per-gate trimming cannot turn a fixed action replay into a general
  trajectory tracker

The rate-limited closed-loop trajectory tracker is now implemented and has
explicit position, velocity, and attitude gain controls. With the ideal v3
trajectory it improved from a best of gate 6 to a best of gate 7, but the
line remained slower and less reliable than the 39.8-39.99 s controller
family. The exact fastsim `FlatRefController` was also tested live and
rejected: it saturated rate commands and reached only gates 2/6/5.

An iterative-learning tool now exists at
`scripts/vq2_adapt_reference_from_live.py`. It learns bounded cross-track
reference corrections from selected-row/live-position logs. Broad and
gate-7-only corrections were tested and rejected; gate-7 speed reduction
reduced tracking error but still did not clear the gate. These experiments
are retained as diagnostics, not promoted artifacts.

## Fast live-policy pivot

The archive contains a faster seed than the original protected checkpoint:

- `D:\ai-gp\training\vq2_sac_runs\vq2_v79_real_rl\20260730_212220\best.pt`
- proven campaign finish: 39.77592 s
- deterministic modern-vision probe: gates 0-9 in about 20 s, then the
  known gate-10 chicane failure

Five deterministic probes produced no finish (deaths at gates 3/5/9/10),
so v79 is fast but marginal and is not promoted over the protected champion.
The corrected g9/g15 map, both alone and with gate-10 gain/lead changes, did
not improve this policy in controlled five-run tests. The active learning
campaign therefore keeps v79's original map/controller and restricts actor
authority and macro exploration to gate 10:

- output: `D:\ai-gp\training\vq2_v93_v79_gate10_macro_awr`
- 12-step AWR, 200 between-episode updates
- 0.008 micro exploration and gate-10-only macro lateral exploration
- modern GPU dense vision plus the crop tracker
- all raw streams recorded under `D:\ai-gp\raw_sessions`

## Promotion ladder

1. Pure deterministic reference/controller evaluation.
2. Candidate must complete a lap before any RL is enabled.
3. Five deterministic regression laps; inspect gate 1, 5, 9, 10, 13, 15,
   and 16 crossing margins.
4. Add bounded residual learning only around a reliable candidate.
5. Compress segment times, starting with gates 11-13, which consume about
   10 seconds of the 39.99-second lap.

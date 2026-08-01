# Optimized reference lines for the SAC campaign (2026-07-31)

Three surrogate-optimized racing lines through the corrected map
(`data/vq2_runtime_map_g9g15fix.json`), delivered in the exact
`vq2_sac_clean_demo.npz` transition format (observation synthesis
validated byte-exact: round-tripping the entire existing clean demo
through the same builder reproduces every channel of every row at 0.0
error; reward uses the v77 constants 2.0/25/0.8/0.02/600).

## Files

| file | cap | planned clearance | clean lap | noisy finish (512 worlds) | noisy median / best |
|---|---|---|---|---|---|
| `data/vq2_lineopt_demo_r1cap8.npz`  |  8 m/s | 0.25 m | 46.30 s | 75.4 % | 47.0 / 42.5 s |
| `data/vq2_lineopt_demo_a1cap10.npz` | 10 m/s | 0.15 m | 39.67 s | 72.7 % | 40.7 / 36.2 s |
| `data/vq2_lineopt_demo_a2cap12.npz` | 12 m/s | 0.15 m | 37.53 s | 67.2 % | 38.5 / 34.2 s |

Reference: the fastest real VQ2 lap ever flown is your teacher's 40.01 s.
The cap-10 line completes clean at 39.7 s and its noisy-median matches
your record while finishing 73 % of heavily randomized worlds.

## How they were made / what the numbers mean

- Search: CEM over per-gate hole-plane crossing offsets (bounded to
  0.75 m − clearance, so the *planned* line never comes closer to the
  hole edge than the stated clearance) + per-segment speed scales.
  32 candidates/round × 14 rounds × 256 worlds each, flown as one
  8192-env batch on the 5090 (`scripts/fastsim_line_opt.py`).
- Every candidate was scored under the full measured noise stack:
  10 Hz-era estimator noise, FOV-coupled fix availability, reloc
  events, 1-step actuation delay, at-rest pitched-pad spawn, obstacle
  cylinders, ±DR on thrust/rates/drag, and your constraint set
  (wire thrust cap 0.52, launch assist 0.30/0.55 s, wire rate limits
  1.348/1.337/0.887).
- The tracker used for scoring is a generic geometric controller
  (`aigp/fastsim/lineopt.py::FlatRefController`) — deliberately dumber
  than your stack (no per-gate tuning, no pins, no crop tracker). The
  "noisy finish" numbers are therefore a floor, not a ceiling.
- Per-gate p05 clearance margins under noise are in
  `data/lineopt/*_report.json` (plus failure histograms and full CEM
  history). Raw winner parameters and the clean trace are in
  `data/lineopt/*_best.npz`.

## Known risk / where the margin is thin

- Gate 1 dominates residual failures on all three lines (the jet+pillar
  precision passage — 98/140 failures on the cap-10 line). p05 noisy
  clearance at g1: 0.075-0.09 m. Your controller's tight g1 tracking
  (47/49 at ±0.3 m) is exactly what these lines assume; my generic
  tracker crosses with more spread.
- The speed schedules trust fastsim_model_v2 aero (fit on 151k real
  samples). If any segment runs hot live, the per-segment speed scales
  in `*_best.npz` (`seg_scale`) can be trimmed without touching the
  geometry.
- Suggested adoption: start with `r1cap8` as a drop-in demo swap to
  validate the pipeline end-to-end (it asks nothing your teacher hasn't
  already done), then move the campaign to `a1cap10`.

## Rebuild / retune

`python scripts/build_lineopt_demo.py --best data/lineopt/a1_cap10_best.npz
 --speed-cap 10 --clearance 0.15 --out <file>` re-derives the npz from
the stored parameters. To re-run or extend the search with different
constraints (other caps, clearances, gate subsets):
`python scripts/fastsim_line_opt.py --speed-cap X --clearance Y
 --out-prefix data/lineopt/tag` on gipsydanger.

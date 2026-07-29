# racer — VQ1 drone-racing RL engine (state-based SAC + MuJoCo twin)

Versioned mirror of the RL racer that lives (and RUNS) in
`C:\Users\satas\Downloads\AI-GP Simulator v1.0.3385-VQ1\PyAIPilotExample-v1\` — the sim
(`FlightSim.exe`) must be launched + logged in manually, and all hardcoded paths
(`racer_state/config.py: run_dir`, the `run_*.bat` launchers) still point at that folder.
Edit here, copy over (or repoint the paths) to run.

## Layout
- `racer_state/` — the REAL-sim (VQ1) side: MAVLink IO (`mav_io.py`, includes the odometry
  pitch-rate sign fix), physical-unit twin→VQ1 action adapter + stack-D trainer (`train2.py`),
  calibration probes (`calibrate.py`, `probe_frames.py`), SAC agent (`sac.py`, `nets.py`),
  dashboards and launchers. `train.py` is the older from-scratch trainer (superseded).
- `racer_mujoco/` — the fast MuJoCo twin: env, vectorized stack-D trainer (`train_mj.py`,
  solved the full 6-gate track 20/20 in 74 min), DAgger vision distillation (`distill.py`),
  recording/dashboard tools.

## Key facts (measured, 2026-07-28)
- VQ1 ODOMETRY `pitchspeed` is sign-flipped vs true rotation (quat-verified); fixed at ingest.
- True cmd→rate gains: roll −2.51, pitch **+2.51**, yaw −2.31; linear to |cmd|≈1.9 (≈5 rad/s).
- Thrust: `specific_force ≈ −3.6 + 62·cmd` m/s², hover ≈ 0.215. Calibration written to
  `runs/calib.json` by `python -m racer_state.calibrate`; `train2.py` maps the policy's
  physical-unit actions (rad/s, thrust-to-weight) through it.
- Gate passes are detected geometrically at the control step (RACE_STATUS is ~4 Hz ⇒ ~3 m late
  at race speed — this smeared credit and blocked greedy consolidation until fixed).

## Not in git
`runs/` and `runs_mj/` (checkpoints, buffers, videos, logs, calib.json) stay on disk in the
Downloads tree. Twin warm-start checkpoint: `racer_mujoco/runs_mj/mj_gate5.pt`.

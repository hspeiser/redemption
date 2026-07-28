# redemption — AI Grand Prix autonomy stack

Vision + control stack for the AI-GP Virtual Qualifier (VQ1), built against the
DCL simulator over MAVLink + UDP camera stream.

## What's here

- `aigp/` — core library
  - `mavlink_io.py` — MAVLink RX/TX hub (full-rate telemetry logging, rate/attitude/motor control, sim reset). Includes the corrected quaternion convention (the sim emits Y-flipped left-handed quats; read as `(w, -x, y, -z)`).
  - `vision_io.py` — chunked-JPEG camera stream receiver (spec §4.6 format).
  - `flight.py` — cascade position→attitude→body-rate controller (command rate capped <100 Hz per spec).
  - `ingest.py` — episode loaders (recorder captures + own logger), strict VQ1/native-timestamp filtering, per-segment sim-clock handling.
  - `calib/` — classical gate detector + ground-truth bundle-adjustment camera calibration.
  - `vision/` — auto-labeling (map projection at ground-truth pose, per-episode clock refinement, orange-mask visibility) and GateNet (corner heatmaps + offsets, pose/velocity/next-gate heads).
- `scripts/` — the working set: dataset build, training (`train_net.py`), live browser overlay with per-gate tracking + multi-gate joint PnP (`live_overlay.py`), replay evaluation against odometry truth (`replay_eval.py`), PnP/blue-robustness/accuracy probes, replay video rendering.

## Measured results (replays of real racing, odometry ground truth)

- Corner localization: ~0.05 px median vs geometric truth (parked), 0.73 px median over the full flight envelope eval
- Gate-relative pose, in motion: **1.4 cm / 0.8° at 0–5 m**, 3.3 cm at 5–10 m
- Absolute pose from multi-gate joint PnP: ~3 cm / ~0.07° (two gates in view)
- Joint-fix availability at race speed: 43–73% per frame (pre-fix baseline: 3–21%)

## Key sim facts (hard-won, verified)

- Gate: 2.70 m outer panel, 1.50 m hole, 0.26 m depth; panel center ≈ 1.07 m above the broadcast map anchor; gates are static.
- Camera: 640×360, fx=fy=320, cx=320, cy=180, pinhole, +20° tilt; verified to sub-0.1 px at rest.
- IMU is noiseless (exact specific force/rates at ~120 Hz).
- ODOMETRY velocity is body-frame; quaternion needs the Y-flip correction; validate attitude conventions on banked frames, never at rest (R == Rᵀ at yaw≈π).

## State estimation (no odometry)

`aigp/ekf.py` — 9-state error-state EKF: position/velocity/attitude from
IMU propagation (~120 Hz) + tightly-coupled gate-corner pixel updates
against the static map. No bias states (sim IMU is measured noiseless).
IMU conventions determined empirically (`scripts/ekf_bringup.py`):
gyro fully negated vs the corrected body frame, accel direct; IMU-only
dead-reckoning drift ≈ 7.6 cm / 0.36° per 1.5 s of banked flight.

`scripts/ekf_replay.py` — full-lap replay grading vs ground truth (used
only for initialization + scoring): mid-race position 5–14 cm median /
~0.3° attitude across full racing episodes, vision+IMU only. Includes
covariance-adaptive corner association and lost-mode relocalization
(pose-head prior + joint PnP re-seed). Known limits: multi-second
look-away vision gaps degrade until relocalization fires.

## VQ2 localization (no odometry, no broadcast map)

Map of record: `data/vq2_map_slam2.json` — built from human corner-fits
(browser editor `scripts/vq2_map_web.py`, click journals in
`data/vq2_map_human*.journal.jsonl`) chained with short IMU hops
(`scripts/vq2_click_chain.py`), plus anchor-based SLAM for late gates
(`scripts/vq2_slam2.py`). The pak-derived `gate_map.json` is off-by-one
(entry 0 = spawn marker; race gate k = entry k+1; gate 17 = finish line,
no physical gate) and carries dm–m per-gate placement error — usable
only as an association prior, never as truth.

Acceptance criteria and status (measured by `scripts/vq2_align.py`,
corner net `data/models/gatenet_v7_best.pt`):

- [x] Independent-lap generalization (lap never used for mapping or
  training, rc_20260724_003101): 2.26 px median innovation, sigma
  3.5 cm median, ~59 % of all frames fusing, <=4 relocalizations.
- [x] Continuous tracking through certified gates: 30 unbroken seconds
  at 1.2–2.1 px / sigma 2–3 cm on the mapping lap; through-gate
  moments at 0.8–1.3 cm.
- [ ] Gates 11/14/16 at click-grade (needs one editor session or a
  looser SLAM observation harvest).
- [ ] sigma p90 < 30 cm full lap (currently 165 cm, dominated by the
  uncertified back-half gates).
- [ ] Racing-speed robustness (fast laps 003405/003525 degrade even on
  certified gates).

## Training

`scripts/train_net.py` — from-scratch recipe: 45 epochs, batch 16, lr 3e-4
cosine, zoom-crop augmentation (pose losses masked on augmented samples),
fp32 focal loss. `--path-map old::new` allows training on a different machine
from the recorder. Champion checkpoint: `data/models/gatenet_v6wsl_best.pt`
(not tracked here by default; see .gitignore).

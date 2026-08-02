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

## Persistent VQ2 pipeline dashboard

The flywheel dashboard remains available when no drone is flying and reads the
archived episode, corpus, world-model, audit, and cycle-ledger artifacts from
`D:\ai-gp`. Start or reuse it with:

```powershell
powershell -ExecutionPolicy Bypass -File .remote\launch_vq2_pipeline_dashboard.ps1
```

Then open `http://127.0.0.1:8900/`. The full-course ABBA launcher also ensures
the dashboard is running automatically. It reports the official record,
recent completion and timing-health rates, late-course reach, eligible training
rows, latest frozen-audit error, record progression, failure gates, recent
flights, and the collect -> ingest -> train -> audit -> decide -> publish state
machine. Every metric includes its interpretation directly in the page.

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

## VQ2 localization: how the generalized vision + EKF system works

The short version is:

> GateNet V7 supplies stable gate/corner identity. A clean-data GateNet V10
> refines only nearby corner pixels without being allowed to invent a gate or
> change its identity. The relative map supplies the matching 3D corners, the
> IMU tracks motion between images, and the EKF combines them into continuous
> local-course pose and velocity.

The generalized runtime uses camera images, gyroscope, accelerometer,
official race-status/gate events, camera calibration, GateNet V7, the
optional V10 corner refiner, and a relative course map. It does **not** load
human click journals, hand-labelled corners, a pose trace from another
flight, VQ1 odometry, VQ2 odometry, per-video handoff times, or future
gate-pass smoothing.

### What the labels were for

There are two different kinds of labels in this project.

**GateNet training labels** teach the neural network which pixels are gate
corners. Every gate has eight corner classes:

- 0–3: inner opening, top-left, top-right, bottom-right, bottom-left.
- 4–7: outside orange panel, in the same order.

Most initial labels were generated automatically in VQ1. Exact VQ1 drone
pose, camera calibration, gate geometry, and gate positions let us project
the 3D gate corners into every image without manually clicking each frame.
GateNet V7 was trained on VQ1 plus the original VQ2 labels; it did **not**
use the later temporal-label set. V8 then added a large V7-self-labelled
image lake, and V9 added temporal labels on top of that. Strict held-out and
human-click evaluation found neither to be a reliable replacement for V7.

GateNet V10 starts from V7 and trains only on clean VQ1 labels, verified VQ2
labels, and verified temporal labels. It has much higher VQ2 candidate recall,
but using it alone can change physical corner classes and destabilize the
existing EKF. The deployed hybrid therefore keeps V7 as the authority and
allows V10 to move an accepted V7 peak by at most 2 pixels, within the same
inner or outer corner ring. V10 cannot add a missing V7 peak, select a gate,
or change the V7 corner class.

**CropGateNet V11** handles the remaining long-range and instance-mixing
problem. A high-recall YOLO model proposes a gate region; the region is padded,
resampled to 256×256, and encoded as RGB, orange likelihood, and a Gaussian
proposal-centre channel. V11 predicts one grouped gate instance:

- Four apparent-image aperture corners in TL, TR, BR, BL order.
- Four grouped outer-panel corners.
- Per-corner visibility and uncertainty.
- Gate-presence confidence.

V11 trained on 77,985 positive crops and 15,597 hard negatives. Validation
uses 30,879 detector-error crop variants from complete held-out sessions.
Training augmentation includes proposal translation/scale errors, motion and
Gaussian blur, noise, exposure changes, clipping, and synthetic pulsing blue
path occlusion.

On 101 independent human-click frames, V11 epoch 13 with robust map-prior
association achieves 90.1% PnP availability and 6.33/12.92/27.88 cm
median/p90/p99 position error. The V7+V10 full-frame champion scores 84.2%
and 10.24/22.71/68.64 cm on the same ruler.

Dense V11 correction is intentionally **not** deployed. Repeated planar PnP
measurements can carry a small systematic bias and drag an otherwise locked
filter. Runtime uses V11 only when the normal V7+V10 corner update fails. A
complete four-corner V11 aperture is solved with PnP, checked against the
projected map gate and inertial continuity, rate-limited, and applied as a
conservative position measurement while gyro attitude remains unchanged.

GateNet has auxiliary global pose, velocity, next-gate, and gate-class heads,
but the current VQ2 localizer does not trust those heads for its world pose.
It uses the network's corner heatmaps and subpixel offsets; the geometric EKF
performs localization.

**Human map labels** are the four hole-corner clicks recorded by
`scripts/vq2_map_web.py`. Known gate dimensions plus those four image points
allow PnP to estimate camera-to-gate translation and possible orientation.
Combined with a trusted camera pose, those measurements helped construct,
diagnose, and verify the relative 3D course map. The clicks are offline map
evidence, not an input required on a new flight.

### Map of record

The runtime map is `data/vq2_map_rawshift.json`. It stores the centre,
full 3D orientation, outside dimensions, and aperture dimensions for race
gates 0–16.

The map was built as follows:

1. Keep the independently measured positions for race gates 0–9.
2. Account for the extracted simulator map's index offset: race gate 0 is
   raw entry 1.
3. Fit one rigid XY rotation/translation and one vertical translation from
   raw entries 1–10 to measured race gates 0–9.
4. Apply that transform to raw entries 11–17 to obtain race gates 10–16.
5. Restore full pitch, yaw, and roll from the extracted simulator
   orientation data.

Full orientation matters. Gate 9 is pitched by roughly 20 degrees; treating
all gates as upright was one source of the old divergence around gate 9.
Race index 17 is a finish marker rather than a physical gate, so it is
excluded from visual fusion and rendering.

### Per-run initialization

Every recording defines a slightly different local frame. Loading a map
literally from the episode that created it rotates or translates the whole
course in a new episode, and a tiny yaw error becomes metres of lateral error
at the back of the course. `scripts/vq2_align.py` therefore initializes every
run independently:

1. Clip camera, IMU, and race-status data to the longest valid monotonic
   simulator-clock segment.
2. Find the period where the drone is parked.
3. Average accelerometer measurements to solve initial roll and pitch from
   gravity; define local yaw as zero.
4. Search the illuminated parked frames for gate 0 with the strict classical
   orange inner/outer-quad detector.
5. Solve gate 0's 3D position from several frames using known gate dimensions,
   camera calibration, and PnP; use the median clean solution.
6. Translate and rotate the entire relative map so its gate 0 matches the
   gate 0 measured in this episode.

Map yaw alignment uses the spawn-to-gate-0 bearing, not the square gate's
solved face orientation. A planar square has nearly equivalent PnP
orientation branches, so trusting its face yaw can inject a few degrees of
error even when its centre translation is accurate.

The drone begins pitched down about 20 degrees while the camera mount
compensates by about 20 degrees, making the image look level. These are still
separate transforms: gravity initializes body attitude and `R_cb` in
`data/calib/calib.json` describes the body-to-camera mounting rotation.

### Stable V7 identity plus V10 corner refinement

Each 640×360 frame becomes a four-channel tensor: normalized RGB plus a soft
orange-likelihood channel. Both networks return eight stride-4 corner
heatmaps and per-class subpixel offsets, producing geometric keypoints rather
than YOLO-style bounding boxes. V7 establishes candidate availability and
corner class. For each accepted V7 peak, V10 may snap its location to the
nearest V10 peak in the same inner/outer ring only when it is within 2 pixels.

At each frame, the EKF projects the mapped 3D corners into the image using its
current pose prediction. The official active-gate event limits association to
the previous, active, and next gate. For each candidate, the matcher:

1. Looks near each projected corner for a V7-authorized peak of the same
   corner class, optionally sharpened by V10.
2. Tests both possible 180-degree square orientations.
3. Keeps the orientation with more matches and lower pixel error.
4. Rejects measurements that are inconsistent with the predicted pose and
   covariance.

Race-status packets are converted to the IMU clock, stably sorted, stripped
of stale pre-reset rows, and forced to progress monotonically. This prevents
late UDP packets from sending association backward and attaching real image
corners to the wrong 3D gate.

### IMU propagation and EKF correction

`aigp/ekf.py` maintains 3D position, velocity, attitude, and covariance.
Every IMU sample propagates the state with body rates, specific force,
gravity, and elapsed simulator time. Camera updates compare each measured V7
corner against the corresponding projected map corner.

The generalized pipeline runs with `--gyro-attitude`. In this mode:

- Gyroscope integration controls attitude.
- Vision corrects position and velocity.
- Vision is not allowed to rotate the nominal attitude.

This separation fixes the characteristic failure where an overlay began in
the correct place and then floated up and right as the drone approached.
Previously, a planar gate's ambiguous orientation could rotate EKF attitude,
rotate gravity incorrectly, and corrupt acceleration, velocity, and position.

### Conservative visual fallback

At extreme range, during motion blur, or when the gate fills the image, V7
may temporarily return too few correctly classified corners. If no V7 corner
update succeeds, the runtime can use a complete V11 aperture observation:

1. Use high-recall YOLO only to propose a padded image crop.
2. Let V11 recover the grouped inner aperture and reject uncertain corners.
3. Require all four corners and robust agreement with the projected map gate.
4. Solve camera-to-gate translation with PnP.
5. Use official race status for gate identity and the map for gate position.
6. Compute body position as
   `p_body = p_gate - R_world_camera @ t_camera_gate`.
7. Keep gyro attitude unchanged.
8. Reject the update if it would cause an implausible jump from
   the inertial prediction.

V11 inference is rate-limited to 10 Hz by default and accepted position pins
are limited to 4 Hz. This fallback handles brief V7 starvation without
allowing a previous or future visible gate to teleport the filter.

### Why sparse sightings can still work

The simulator IMU has very low noise and bias. A gate observation corrects
position and velocity, the IMU carries the estimate through a short visual
gap, and the next gate observation removes accumulated drift. The architecture
can therefore remain localized without detecting a gate on every frame,
although the current replays fuse V7 corners much more frequently than once
per 30 frames.

Current label-free replay validation using V7+V10 with sparse V11 fallback on
two different VQ2 recordings:

- Clean no-contact run: 93.2% of frames fused corners, 2.56 px median
  innovation, 3.3 cm median / 6.2 cm p90 reported position sigma. V7 alone
  fused 87.5% with 4.7/9.0 cm sigma.
- Independent `rc_20260724_003101` run: 82.8% fused, 4.00 px median
  innovation, 4.4 cm median / 8.7 cm p90 sigma. V7 alone fused 76.8% with
  5.1/12.5 cm sigma.

The 10 Hz sparse fallback fired only 6 times on the clean lap and 4 times on
`003101`; V7+V10 remains authoritative on every normal frame. Human clicks
are used only for evaluation, never as runtime filter inputs.

These covariance values are filter confidence rather than absolute
ground-truth guarantees, so visual overlays and independent click residuals
remain part of acceptance testing.

### Runtime data flow

```text
camera + IMU + official race status
                 |
                 v
valid monotonic clock segment
                 |
                 v
gravity attitude + parked gate-0 PnP
                 |
                 v
relative map re-anchored to this episode
                 |
                 v
IMU continuously propagates pose and velocity
                 |
                 v
GateNet V7 predicts stable candidate identity
                 |
                 v
V10 may refine each accepted location by <=2 px
                 |
                 v
race status narrows map association
                 |
                 v
matched map/image corners correct EKF position and velocity
                 |
                 v
YOLO proposal + V11 aperture PnP bridges rare V7 dropouts
                 |
                 v
continuous local-course pose, velocity, and uncertainty
```

The same model and estimator can process another recording without
annotating it, provided the course map is correct, gate 0 is visible while
parked, camera calibration is unchanged, timestamps are valid, and official
race-status messages are available. A different gate arrangement requires a
different relative map, not a retrained corner detector.

## Training

`scripts/train_net.py` — from-scratch recipe: 45 epochs, batch 16, lr 3e-4
cosine, zoom-crop augmentation (pose losses masked on augmented samples),
fp32 focal loss. `--path-map old::new` allows training on a different machine
from the recorder. The VQ2 corner/EKF pipeline keeps
`data/models/gatenet_v7_best.pt` as the identity authority and optionally
uses the clean-data V10 checkpoint as a tightly bounded location refiner:

```powershell
.venv-train\Scripts\python.exe scripts\vq2_align.py `
  --episode-dir <episode> `
  --map-json data\vq2_map_rawshift.json `
  --ckpt data\models\gatenet_v7_best.pt `
  --corner-refine-ckpt data\models\gatenet_v10strict_ep0.pt `
  --corner-refine-radius 2 `
  --crop-gatenet-ckpt data\models\crop_gatenet_v11crop_ep13.pt `
  --crop-proposal-ckpt data\models\gatepose_v5vq2b_best.pt `
  --crop-proposal-thresh 0.05 `
  --crop-proposal-padding 2.6 `
  --crop-v11-position-pins `
  --crop-v11-inference-interval 0.10 `
  --crop-v11-pin-interval 0.25 `
  --thresh 0.12 `
  --gyro-attitude
```

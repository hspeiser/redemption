# Vision Hand-off: Localization Limits & What the Vision System Needs to Improve

**Audience:** the perception / `gate_nano` team.
**TL;DR:** The state estimator (EKF) is not the bottleneck — given a good position
measurement it localizes to **~3 cm**. The real `gate_nano` detector limits us to
**~26 cm** in flight, driven by the **monocular depth ambiguity of a single planar gate**:
the raw per-frame position measurement is a fine **12 cm median** but has a **7 m p90 tail**.
Closing that tail is a vision-side problem. This document quantifies it and lists concrete,
prioritized asks.

---

## 1. System context (how vision feeds localization)

Pipeline per camera frame:

```
FPV frame (640×360) ─► gate_nano (YOLO11-Pose) ─► 4 inner-corner keypoints + confidences
                                                        │
                                     gravity-constrained PnP (upright solver)
                                                        │
                                    gate pose in camera frame  ─►  EKF (fuses with IMU)
```

- **Camera** (matches training): pinhole, 640×360, `fx=fy=320`, `cx=320, cy=180`,
  no distortion. HFoV 90°, VFoV 58.7°.
- **Gate**: square, inner opening **1.5 m**, outer 2.7 m, tube depth 0.26 m. The detector
  outputs the **4 inner corners** (TL, TR, BR, BL) each with an (x, y, confidence).
- **Estimator**: error-state EKF, IMU strapdown at 250 Hz + gravity-aided attitude,
  gate measurements at ~30–48 Hz. It already does outlier gating, robust init, and
  gravity leveling (see §5).

---

## 2. The measured limit (the core evidence)

We ran the identical trajectory through three measurement sources to separate **estimator
quality** from **vision quality** (`scripts/eval_ekf.py`):

| Measurement source | Drone position RMS (sustained flight) |
|---|---|
| **EKF + ideal oracle** (true position + small noise) | **3.0 cm** |
| EKF + **real `gate_nano`** | **26.6 cm** |
| EKF + real detector + known track map | 24.9 cm |
| EKF + known map, **stationary hover** | 4.5 cm |

The estimator reaches **3 cm** with a good measurement → **the EKF is not the limit.**

We then measured the **raw per-frame position error** of the vision measurement itself
(the floor no filter can beat), over a realistic racing trajectory (`scripts/measure_vision.py`):

| Corner source | median | **p90** | max |
|---|---|---|---|
| Synthetic corners (2 px noise) | 2.0 cm | 12.8 cm | 46 cm |
| **Real `gate_nano`** | 12.2 cm | **717 cm** | 733 cm |

**The median is good (12 cm). The problem is the heavy tail: 10% of measurements are off
by >7 meters.** Those blowups, even when mostly gated out by the EKF, are what cap accuracy.

Corner-localization accuracy vs. range (`scripts/check_detector.py`, per-corner pixel error
vs. ground-truth projection):

| range | box conf | keypoint conf | max corner error |
|---|---|---|---|
| 5 m | 0.6 | ~1.0 | **3.8 px** (excellent) |
| 3 m (oblique) | 0.68 | ~1.0 | ~15 px |
| 2 m (gate fills frame) | 0.81 | ~0.9 | ~26 px |
| **1 m (gate overflows frame)** | — | ~1.0 | **→ upright solve gives 5.8 m position error** |

---

## 3. Root-cause analysis

### 3a. Monocular depth ambiguity of a planar square (dominant)
Range from a single planar gate comes from its **apparent size**: `depth ≈ f · gate_size /
apparent_pixel_size`. Corner-pixel noise therefore maps to a *proportional range error*:

```
depth_error ≈ (corner_pixel_noise / apparent_gate_pixels) × range
            ≈ (10 px / 120 px) × 4 m  ≈  0.33 m at 4 m   (≈ our observed ~26 cm)
```

Near fronto-parallel views the perspective signal is weakest, so small pixel errors produce
**large, occasionally catastrophic** range errors (the 7 m tail). Bearing (lateral/vertical)
stays accurate; **depth is the failure axis.**

### 3b. Confidence is not calibrated to error
`gate_nano` reports keypoint confidence ~0.9–1.0 **even on the detections that yield 7 m
position errors.** The estimator has no signal to down-weight or reject a bad-depth frame,
so it must rely on statistical gating after the fact.

### 3c. Close-range / truncated gates blow up
When the drone is within ~1.4 m the gate overflows the frame; corners land at/over the image
border and the pose solve diverges (5.8 m error at 1 m). The estimator currently just drops
these frames (coasts on IMU), which is safe but loses vision exactly when you're threading a gate.

### 3d. Only inner corners are used
The model detects **4 inner corners**. A square is minimally constrained; adding the **4 outer
corners** (8-point PnP) would materially improve pose conditioning and depth observability.

### 3e. Sim→real domain gap (flagged, not yet measured)
`gate_nano` was trained on the redemption synthetic renderer; it detects our MuJoCo gates well
(conf 0.5–0.8) but corner accuracy degrades up close. Real-hardware imagery (motion blur,
lighting, real gate texture/orange rim) will differ — expect to re-validate/fine-tune.

---

## 4. What the vision side should improve (prioritized)

**P0 — Calibrated per-detection uncertainty.** Output a **per-corner 2×2 pixel covariance**
(or at least a calibrated scalar) that *actually correlates with error*. This single change
lets the estimator weight and gate correctly and would neutralize most of the 7 m tail.
Success metric: reported σ predicts observed corner error (reliability curve ≈ diagonal).

**P0 — Reduce corner pixel noise.** Depth error is *linear* in corner noise, so halving pixel
error halves range error. Levers: higher input resolution / a sub-pixel refinement head,
heavier/better-augmented training, test-time augmentation, or a larger backbone if latency
allows. Success metric: corner RMS < 3 px out to 6 m (currently ~4 px at 5 m, ~15–26 px close).

**P1 — Detect the outer corners too (8 keypoints).** Inner+outer corners tighten PnP depth
and add redundancy for outlier detection. Nearly free to add to the pose head.

**P1 — Flag truncated / partial gates.** Emit an `overflow`/`truncated` flag (any corner
outside the image, or gate subtends > ~70% of frame) so the estimator can trust or drop the
frame explicitly instead of guessing. Bonus: a model that localizes robustly from a *partial*
gate would restore vision during close fly-throughs (§3c).

**P2 — Explicit depth cue.** The gate has a 0.26 m tube; detecting the tube/side faces or the
back-face corners gives a direct depth cue that breaks the planar ambiguity. Alternatively a
monocular metric-depth prior on the gate region.

**P2 — Encourage multi-gate detection.** When ≥2 gates are visible, the estimator can
triangulate depth. Make sure the detector reliably returns *all* in-frame gates, not just the
nearest/largest, with stable IDs.

**P3 — Temporal consistency.** A lightweight tracker (or learned VO over the gate) that fuses
corners across frames would cut the tail — though note the EKF already does temporal fusion, so
the highest-value vision work is per-frame uncertainty + noise (P0), not re-implementing filtering.

---

## 5. What the estimator already does (so you don't rebuild it)

- IMU strapdown @ 250 Hz + **gravity-aided attitude** (accelerometer levels roll/pitch) — bounds
  attitude drift between gates.
- **Gravity-constrained upright PnP** front-end — removes the planar-square *rotation* mirror
  ambiguity (but NOT the depth ambiguity).
- **Chi²-gated** updates + edge/range gating — rejects the worst blowups (this is what keeps the
  7 m tail from being fatal; it is a mitigation, not a fix).
- Candidate-confirmed landmark init (no ghost gates); optional known-track anchoring (`map` mode).
- Tuned so depth is trusted about as much as bearing; loosening depth was tried and **hurt**
  (IMU-only depth drifts), so we do need vision depth — just less noisy.

**Net:** every remaining lever we found is on the vision side. The estimator is at its ceiling.

---

## 6. How to measure improvement (reuse our harness)

- `scripts/measure_vision.py` — **primary metric.** Reports raw per-frame position-measurement
  error (median / **p90** / max). Target: drive p90 from 7 m toward <30 cm.
- `scripts/check_detector.py` — per-corner pixel error vs. range; saves annotated frames.
- `scripts/eval_ekf.py [racing|flythrough]` — end-to-end EKF RMS, with `synth_vision` (ceiling)
  vs `real` (current) side by side. Target: close the 26 cm → 3 cm gap.

A new detector build is "better" iff `measure_vision.py` p90 drops and `eval_ekf.py real`
approaches `synth_vision`.

---

## 7. Interface contract (what the estimator wants from vision)

Per frame, per detected gate:

| field | now | requested |
|---|---|---|
| 4 inner corners (px) | ✅ | keep |
| per-corner confidence | ✅ (uncalibrated ~1.0) | **calibrated per-corner σ or 2×2 covariance (P0)** |
| box confidence | ✅ | keep |
| outer corners (px) | ✗ | **add (P1)** |
| truncated/overflow flag | ✗ | **add (P1)** |
| stable gate ID | ✗ | nice-to-have (helps multi-gate association) |
| (optional) gate 6-DoF pose + full covariance | ✗ | if you solve pose in-network, return an honest covariance (large along depth) |

If you provide a calibrated covariance, the estimator can consume it directly (it already
supports anisotropic measurement noise); no estimator changes needed on your account beyond
wiring the fields.

---

## Appendix — fixed parameters

- Camera: 640×360, fx=fy=320, cx=320, cy=180, dist=0. HFoV 90°, VFoV 58.7°.
- Gate: inner 1.5 m, outer 2.7 m, tube depth 0.26 m; corners ordered TL, TR, BR, BL; color #fe3201.
- Detector: `gate_nano_best.pt` (Ultralytics YOLO11-Pose, 1 class, 4 keypoints, imgsz 640).
- Estimator config: `configs/app.toml [ekf]` / `[localization]`.
- Current end-to-end: hover 4.5 cm (map), flight ~26 cm (real) vs 3 cm (oracle ceiling).

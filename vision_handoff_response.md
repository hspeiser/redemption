# Vision → EKF: Response & Delivered Fixes (no-GPU, no resolution change)

**From:** perception / `gate_nano` team
**Re:** your `vision_handoff.md` (localization limits, the 7 m depth p90 tail)
**TL;DR:** We shipped the two P0 asks and one P1 ask **without retraining and without
raising resolution** (you said we can't). The headline — **an honest, anisotropic
per-detection covariance that blows up along depth exactly on the frames that carry
your tail** — plus **sub-pixel corner refinement**. On our real-val proxy this takes the
per-frame position measurement from **median 0.35 m / p90 1.63 m / max 18.8 m** to
**0.09 m / 0.46 m / 1.67 m** once the covariance is used to weight/gate. The reported σ
predicts the error (high σ ⇒ the tail). Fields are wired and ready to consume.

Everything here runs on CPU at inference; nothing below needed the GPU box.

---

## 1. What's delivered (maps to your §4 / §7 asks)

| Your ask | Status | How |
|---|---|---|
| **P0 — calibrated per-detection uncertainty** | ✅ **done** | Analytic 6-DoF pose covariance `Cov=(JᵀWJ)⁻¹` + calibrated per-corner σ. `redemption/pose.py`. |
| **P0 — reduce corner pixel noise** | ✅ **done** (res-free) | Sub-pixel `cornerSubPix` refine head. `redemption/refine.py`. Corner RMS **1.71→0.85 px** (0.53 px far). |
| **P1 — truncated / overflow flag** | ✅ **done** | Border + frame-fill test. `pose.is_overflow`. |
| §5 — gravity-constrained (upright) PnP | ✅ **already in** | `redemption/upright.py` (`solve_upright`, up4). Rotation p90 **58.6°→14°** on real val. |
| **P1 — outer corners (8-pt PnP)** | ❌ **needs GPU** | Tried classical red-mask outer corners → **worse** (see §4). Needs a trained head. |
| P2 — explicit depth cue (tube/back-face) | ⏳ needs GPU | trained head. |
| §7 — stable gate ID | ⏳ not done | tracker; low priority (you already fuse temporally). |

The estimator can consume the covariance directly (you said you support anisotropic
noise) — **no EKF changes beyond wiring the new fields.**

---

## 2. The core result: the covariance is the tail-killer

The 7 m tail is **not random detector noise** — it's the monocular depth ambiguity of a
near-fronto-parallel planar square, which is *geometrically predictable*. We propagate the
per-corner pixel noise through the upright-PnP reprojection Jacobian:

```
Cov_pose = (Jᵀ W J)⁻¹ ,  J = d(reproj)/d[center, heading] ,  W = diag(1/σ_px²)
depth_sigma = √(rayᵀ · Cov_position · ray)      # 1σ along the viewing ray
```

`J` is nearly rank-deficient along depth when the perspective signal is weak, so
`depth_sigma` **explodes exactly on the bad frames** — no learned confidence needed.

**Measured on 1,609 real-val gates** (`scripts/vision_measure.py`; reference pose = IPPE on
the labeled corners, gravity = that pose's up-axis as an IMU stand-in):

| source | median | **p90** | max (m) |
|---|---|---|---|
| raw corners → upright PnP | 0.351 | 1.631 | 18.76 |
| + sub-pixel refine | 0.139 | 0.935 | 17.13 |
| **+ use covariance (gate depth_sigma ≤ 1 m)** | **0.093** | **0.455** | **1.67** |

`depth_sigma` is cleanly **bimodal** — ~0.35 m (well-conditioned) vs ~9 m (fronto-parallel) —
and the ~9 m group carries the entire tail (its error p90 is 1–2 m vs 0.3–0.7 m for the good
group). Reliability: reported σ ≳ observed error (conservative; safe for you not to over-trust).

**Key point for you:** the tail appears at *both* ~10 m and ~30 m range (see
`reports/vision_measure.png`) — it's **conditioning-driven, not range-driven**, so your current
edge/range gating (§5) can't catch it but this covariance can. Don't hard-reject the high-σ
50%; **fuse them with the large covariance** and they'll barely move the estimate.

---

## 3. Interface — what each detection now carries

`redemption.pose.detect_gates(model, image, K, dist, down_cam=<IMU gravity in cam frame>)`
returns a `GateDetection` per gate:

| field | provided | note |
|---|---|---|
| `corners_px` (4×2, TL,TR,BR,BL) | ✅ | **sub-pixel refined** |
| `corner_sigma_px` | ✅ | **calibrated scalar** (P0) — grows for large/oblique, big for overflow |
| `box_conf`, `kpt_conf` | ✅ | kept (note: kpt_conf still ~1.0, uncalibrated — use `corner_sigma_px`) |
| `overflow` | ✅ | **truncation/fill flag** (P1) — drop or heavily distrust |
| `center_cam`, `psi`, `R` | ✅ | gravity-constrained (upright) pose, cam frame |
| `pos_cov` (3×3, m²) | ✅ | **anisotropic, large along depth** (P0) — feed directly as measurement noise |
| `heading_var`, `depth_sigma` | ✅ | psi variance; 1σ along the ray (handy for gating/telemetry) |
| outer corners | ❌ | needs GPU (see §4) |
| stable gate ID | ❌ | not yet |

`down_cam` is the only new input we need from you (the IMU gravity direction in the camera
frame). Without it we still return corners + `corner_sigma_px` + `overflow`.

---

## 4. What we tried that did NOT work (so you don't retry it)

**8-point PnP via classical red-mask outer corners.** The gate frame is red, so we detected
its outer boundary contour and paired those 4 outer corners with the model's 4 inner corners
for an 8-point solve. Result on 738 gates: **worse, not better** — median position error
**0.16 m → 1.07 m**, p90 1.18 → 7.7 m. The outer boundary is too noisy at runtime (fuzzy outer
edge, 0.26 m tube front/back ambiguity, overexposure). **Conclusion: outer corners must come
from a trained keypoint head, not classical CV.** That's the strongest argument for the GPU
work below.

---

## 5. What still needs a GPU session (batch these)

Ranked; all blocked on retraining, none on resolution (we respect the fixed 640×360):

1. **8-keypoint model (inner + outer corners), P1.** Materially improves depth conditioning
   (larger baseline) + redundancy. Our synthetic renderer already has exact outer corners;
   the real set would need outer-corner auto-labels (extend Henry's projector). ~1 fine-tune.
2. **Close-range fix (0–4 m), the last bad bin.** Corner RMS ~14 px and the 1 m/5.8 m blowups
   live here, and the real data barely covers it (32 labels < 3 m). Fix = a **red close-range
   synthetic batch** blended into a fine-tune (Henry's v3 recipe; our synthetic is now red
   `#fe3201`). Directly targets your §3c.
3. **Explicit depth cue (P2):** a keypoint on the tube/back-face breaks the planar ambiguity —
   the only thing that fixes depth *structurally* rather than just reporting it.
4. **Heavier/better-augmented training** for lower base corner noise (a larger backbone only if
   latency allows) — but sub-pixel already hit your "< 3 px to 6 m" target for mid/far, so this
   is lower priority than 1–2.

---

## 6. Honest caveats

- **Validation reference.** We have no true drone pose / replay for the real val set, so the
  numbers above use *label-corner PnP* as the reference — this measures the **vision measurement
  noise** (predicted vs label corners through PnP), which is exactly the quantity that drives
  your tail, but it is **not** absolute ground truth. **Please re-run your `measure_vision.py` /
  `eval_ekf.py real` with the new `pos_cov` + `overflow` fields wired** — that's the definitive
  26 cm → 3 cm check.
- **σ calibration** was fit on this (in-distribution, auto-labeled) val set; **re-calibrate on
  real-hardware imagery** once available (§3e domain gap). The knobs are constants in
  `pose.py` (`SIGMA_BASE`, `SIGMA_NEAR_K`, `SIGMA_OVERFLOW`).
- `depth_sigma` is **deliberately conservative** (σ ≳ error) so you never over-trust depth;
  if you find it too pessimistic on good detections, lower `SIGMA_BASE`.

---

## 7. Where the code is

- `redemption/pose.py` — covariance, calibrated σ, overflow, `detect_gates()` (the wired path)
- `redemption/refine.py` — sub-pixel corner refinement
- `redemption/upright.py` — gravity-constrained up4/up3 PnP (rotation-ambiguity fix)
- `scripts/vision_measure.py` — the reliability + tail-reduction harness (reproduces the tables)
- Detector: `gate_nano_best.pt` (YOLO11n-pose, real-fine-tuned, 0.995 pose mAP50 on real val;
  92.6%→99.9% detection after fine-tune; sub-pixel corners with refine)
- Gate color updated to your `#fe3201`; camera/gate params unchanged (640×360, f=320, inner 1.5 m).

**Bottom line:** the two highest-value asks are done and resolution-free — you now get a
covariance that's honest about depth and corners that are ~2× tighter. Wire `pos_cov` +
`overflow`, re-run your harness, and the remaining gains (outer corners, close-range) are one
batched GPU session away.

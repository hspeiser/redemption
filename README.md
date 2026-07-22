# Redemption — Synthetic YOLO-Pose Gate-Corner Detection + PnP

Train a **YOLO-Pose** model to detect the **4 inner corners** of bright-orange
square racing gates in 640×360 pinhole-camera images, then recover each gate's
**6-DoF pose** with a confidence-weighted **PnP** solver — all from procedurally
generated synthetic data.

Everything is driven by **TOML config files** in `configs/` — there are **no
command-line arguments** anywhere.

---

## Pipeline

```
 generate_data  ->  distribution plots  ->  train (YOLO-Pose)  ->  PnP eval  ->  reports
```

1. **Data generation** (`scripts/generate_data.py`) — render up to 4 gates per
   image at random uniform positions (sub-meter … 15 m) and pitch/yaw, over
   procedural dark backgrounds with domain randomization. Only **fully-visible**
   gates (all 4 inner corners in-frame & unoccluded) are labelled. Writes an
   Ultralytics YOLO-Pose dataset plus ground-truth pose sidecars.
2. **Distribution plots** — 3D scatter of gate centers, histograms + KDE of
   depth/pitch/yaw/position/size, a corner-coverage heatmap, and KS-test
   uniformity checks, so you can confirm the data is well-distributed.
3. **Training** (`scripts/train_model.py`) — Ultralytics training with solid
   checkpointing (`best`/`last` + periodic `epochN.pt`) and full loss/metric
   logging. Emits loss/metric curves and predicted-vs-GT overlay montages.
4. **PnP evaluation** (`scripts/evaluate_pnp.py`) — for every checkpoint, run
   inference → solve PnP (IPPE) with confidence-weighted refinement → compare to
   ground truth. Plots **reconstruction error vs training checkpoint** plus a
   confidence-vs-error scatter.
5. **Reports** — every data-gen and training run auto-generates a timestamped,
   self-contained report in `reports/` with plots and 🟢/🟡/🔴 success indicators.

---

## Setup

Requires an NVIDIA GPU. This project pins **Python 3.11** and pulls the CUDA
12.4 PyTorch wheels (forward-compatible with newer drivers).

```bash
uv sync
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # expect True
```

## Run

```bash
# Full pipeline end-to-end:
uv run python scripts/run_all.py          # (or: uv run python main.py)

# Or stage by stage:
uv run python scripts/generate_data.py
uv run python scripts/plot_distribution.py
uv run python scripts/train_model.py
uv run python scripts/evaluate_pnp.py
uv run python scripts/make_report.py      # rebuild reports from existing artifacts
```

## Staged model scaling (recommended)

Data quality matters more than model size, so iterate on the data first with the
**nano** model, then scale up by editing `configs/train.toml`:

| Stage | `[model].weights`  | Notes                            |
| ----- | ------------------ | -------------------------------- |
| POC   | `yolo11n-pose.pt`  | fastest, lowest VRAM (default)   |
| Scale | `yolo11s-pose.pt`  | better accuracy                  |
| Final | `yolo11m-pose.pt`  | highest ceiling, more VRAM/time  |

## Fast smoke test

To validate the whole pipeline in a couple of minutes, temporarily shrink
`configs/datagen.toml` (`n_train=40, n_val=8, n_test=12`) and `configs/train.toml`
(`epochs=3`, `batch=8`, `save_period=1`), then run `scripts/run_all.py`.

---

## Configuration (`configs/`)

| File           | Controls                                                            |
| -------------- | ------------------------------------------------------------------- |
| `camera.toml`  | Intrinsics `K`, resolution — the **source of truth** for projection |
| `gate.toml`    | Gate geometry (outer 2700, inner 1500, depth 260 mm), colour        |
| `datagen.toml` | Counts, splits, pose ranges, visibility filter, backgrounds, augment |
| `train.toml`   | Model size, epochs, batch, device, checkpointing, augmentation      |
| `pnp.toml`     | Solver method, confidence thresholds/weighting, checkpoint selection |
| `report.toml`  | Report format, plot toggles, success thresholds                     |

### Note on the camera spec
`size_details.md` lists `VFoV = 90°`, but the intrinsics (`fy=320, cy=180`) give
a **vertical** FoV of ≈58.7° and a **horizontal** FoV of exactly 90°. The "90°"
is the horizontal field of view; the code trusts the intrinsics matrix `K`.

---

## Layout

```
configs/           TOML configuration (all behaviour)
src/redemption/    library: camera, gate, geometry, render, datagen, train, pnp, report ...
scripts/           thin entrypoints (no argparse)
datasets/          generated datasets (gitignored)
runs/              Ultralytics training runs + checkpoints (gitignored)
reports/           generated reports (gitignored)
```

## Coordinate & corner conventions
- Camera frame: +X right, +Y down, +Z into the scene (OpenCV).
- Gate pose maps object→camera: `X_cam = R @ X_obj + t`, with `R` from pitch
  (about +X) and yaw (about +Y); roll = 0.
- Inner corners are ordered **TL, TR, BR, BL** (index tied to the physical
  corner, not the view) for both keypoints and PnP correspondences.

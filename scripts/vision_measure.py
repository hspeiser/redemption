"""Vision self-check: per-frame position-measurement error + uncertainty reliability.

Mirrors the EKF team's `measure_vision.py` intent on our real val set, to prove the
resolution-free fixes (sub-pixel refine + analytic covariance gating + overflow flag)
shrink the p90 depth tail and that the reported sigma predicts the error.

Reference pose per gate = IPPE on the LABELED corners; gravity = that pose's up axis
(stands in for the IMU). Measurement = predicted-corner upright-PnP position; error =
||t_pred - t_ref||. No GT drone pose / replay needed. Config-free eval.

    uv run python scripts/vision_measure.py <ckpt> <val_dir> [max_images]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from redemption.camera import PinholeCamera  # noqa: E402
from redemption.config import load_toml  # noqa: E402
from redemption.infer import infer_image, load_model  # noqa: E402
from redemption.metrics import match_by_center  # noqa: E402
from redemption.pnp import solve_pnp  # noqa: E402
from redemption.pose import (apparent_size_px, corner_sigma, depth_sigma_m,
                             is_overflow, pose_covariance)  # noqa: E402
from redemption.refine import refine_corners  # noqa: E402
from redemption.upright import solve_upright  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/gate_nano_best.pt"
VAL = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
    "C:/Users/satas/AppData/Local/Temp/claude/"
    "C--Users-satas-projects-redemption/80309095-61c5-4563-bb84-c8fa96cf71e2/scratchpad/real_val")
MAXN = int(sys.argv[3]) if len(sys.argv) > 3 else 0

cam = PinholeCamera.from_config(load_toml("camera.toml"))
K, dist, W, H = cam.K, cam.dist_coeffs, cam.width, cam.height
HALF = 0.75
OBJ = np.array([[-HALF, HALF, 0.], [HALF, HALF, 0.], [HALF, -HALF, 0.], [-HALF, -HALF, 0.]])
model = load_model(CKPT)


def parse(txt):
    g = []
    for ln in txt.strip().splitlines():
        p = ln.split()
        if len(p) >= 17:
            g.append(np.array([[float(p[5 + i * 3]) * W, float(p[5 + i * 3 + 1]) * H]
                               for i in range(4)], float))
    return g


rows = []  # dicts per matched gate
imgs = sorted(VAL.glob("images/*.jpg"))
if MAXN:
    imgs = imgs[:MAXN]
for n, ip in enumerate(imgs):
    lp = VAL / "labels" / (ip.stem + ".txt")
    if not lp.exists():
        continue
    img = cv2.imread(str(ip))
    if img is None:
        continue
    gts = parse(lp.read_text())
    if not gts:
        continue
    dets = infer_image(model, img, conf=0.25, imgsz=640, device=0)
    if not dets:
        continue
    pc = np.array([d.center_px for d in dets])
    gc = np.array([g.mean(0) for g in gts])
    for pi, gi in match_by_center(pc, gc, 60.0):
        d, gt = dets[pi], gts[gi]
        sgt = solve_pnp(OBJ, gt, K, dist, "IPPE", True, False, np.ones(4))
        if sgt is None:
            continue
        t_ref, R_ref = sgt["tvec"], sgt["R"]
        rng = float(np.linalg.norm(t_ref))
        down_cam = -R_ref[:, 1]
        raw = np.asarray(d.kpts_px, float)
        ref = refine_corners(img, raw)
        rec = {"rng": rng, "overflow": is_overflow(ref, W, H),
               "corner_err": float(np.sqrt(np.mean(np.sum((ref - gt) ** 2, axis=1))))}
        for tag, cor in (("raw", raw), ("ref", ref)):
            su = solve_upright(cor, down_cam, K, HALF, weights=d.kpt_conf)
            rec[f"err_{tag}"] = np.linalg.norm(su["tvec"] - t_ref) if su else np.nan
            if tag == "ref" and su is not None:
                ap = apparent_size_px(cor)
                sig = corner_sigma(ap, rec["overflow"])
                cov = pose_covariance(su["center"], su["psi"], down_cam, K, HALF, sig)
                rec["sigma_px"] = sig
                rec["depth_sigma"] = depth_sigma_m(su["center"], cov[:3, :3])
        rows.append(rec)
    if (n + 1) % 300 == 0:
        print(f"  {n+1}/{len(imgs)}", flush=True)

R = rows
er = np.array([r["err_raw"] for r in R], float)
ef = np.array([r["err_ref"] for r in R], float)
ds = np.array([r.get("depth_sigma", np.nan) for r in R], float)
ov = np.array([r["overflow"] for r in R], bool)
ce = np.array([r["corner_err"] for r in R], float)


def stats(a):
    a = a[np.isfinite(a)]
    return (np.median(a), np.percentile(a, 90), np.max(a), len(a))


print(f"\nmatched gates: {len(R)}\n")
print(f"{'source':>28} | {'median':>8} {'p90':>8} {'max':>8} (m)")
print("-" * 62)
for name, a in (("raw corners -> upright", er),
                ("+ sub-pixel refine", ef)):
    m, p, mx, nn = stats(a)
    print(f"{name:>28} | {m:8.3f} {p:8.3f} {mx:8.2f}")

# --- overflow flag: drop flagged detections ---
keep_of = ~ov
m, p, mx, nn = stats(ef[keep_of])
print(f"{'+ drop overflow-flagged':>28} | {m:8.3f} {p:8.3f} {mx:8.2f}   "
      f"(dropped {ov.sum()}/{len(ov)} = {100*ov.mean():.1f}%)")

# --- covariance gating on depth_sigma ---
for tau in (1.0, 0.5, 0.3):
    keep = np.isfinite(ds) & (ds <= tau) & (~ov)
    a = ef[keep]
    a = a[np.isfinite(a)]
    if len(a):
        rej = 100 * (1 - len(a) / np.isfinite(ef).sum())
        print(f"{'+ gate depth_sigma<=' + str(tau) + 'm':>28} | "
              f"{np.median(a):8.3f} {np.percentile(a,90):8.3f} {np.max(a):8.2f}   "
              f"(reject {rej:.1f}%)")

# --- reliability: does predicted depth_sigma predict the actual position error? ---
valid = np.isfinite(ds) & np.isfinite(ef)
x, y = ds[valid], ef[valid]
if len(x) > 20:
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    nb = 8
    edges = np.quantile(xs, np.linspace(0, 1, nb + 1))
    print(f"\nreliability (predicted depth_sigma -> observed error):")
    bx, by = [], []
    for i in range(nb):
        m = (xs >= edges[i]) & (xs <= edges[i + 1])
        if m.sum() >= 3:
            bx.append(np.median(xs[m]))
            by.append(np.median(ys[m]))
            print(f"  sigma~{np.median(xs[m]):6.3f}m -> err median {np.median(ys[m]):6.3f}m "
                  f"p90 {np.percentile(ys[m],90):6.3f}m  (n={m.sum()})")
    corr = np.corrcoef(np.log10(x + 1e-3), np.log10(y + 1e-3))[0, 1]
    print(f"  log-log corr(sigma, error) = {corr:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].loglog(x, y, ".", alpha=0.25, ms=3)
    if bx:
        ax[0].loglog(bx, by, "o-", color="crimson", label="binned median")
    lim = [min(x.min(), y.min()) + 1e-3, max(x.max(), y.max())]
    ax[0].loglog(lim, lim, "k--", alpha=0.5, label="ideal (y=x)")
    ax[0].set_xlabel("predicted depth sigma (m)"); ax[0].set_ylabel("observed position error (m)")
    ax[0].set_title("Uncertainty reliability"); ax[0].legend(fontsize=8)
    # error vs range, colored by overflow
    rr = np.array([r["rng"] for r in R])
    ax[1].semilogy(rr[~ov], ef[~ov], ".", alpha=0.3, ms=3, label="ok")
    ax[1].semilogy(rr[ov], ef[ov], ".", alpha=0.5, ms=4, color="red", label="overflow-flagged")
    ax[1].set_xlabel("range (m)"); ax[1].set_ylabel("position error (m)")
    ax[1].set_title("Error vs range"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    out = Path("reports") / "vision_measure.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nplot -> {out}")

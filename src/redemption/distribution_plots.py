"""Dataset distribution analysis + matplotlib plots.

Reads the ground-truth sidecars for a split and produces:
  * a 3D scatter of gate centers (camera frame), colored by yaw,
  * histograms + KDE for depth, pitch, yaw, image-plane x/y and apparent size,
  * a keypoint-coverage heatmap over the 640x360 frame,
  * a gates-per-image bar chart,
plus KS-test uniformity p-values used by the report's success indicators.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from .config import DotDict, load_all  # noqa: E402
from .dataset import dataset_root, split_paths  # noqa: E402
from .utils import ensure_dir, get_logger, read_json  # noqa: E402


def collect_stats(root: Path, split: str) -> dict:
    """Gather per-gate arrays from a split's meta sidecars."""
    meta_dir = split_paths(root, split)["meta"]
    depth, pitch, yaw = [], [], []
    cx, cy, cz = [], [], []
    area, gates_per_image = [], []
    corner_x, corner_y = [], []

    for mf in sorted(meta_dir.glob("*.json")):
        meta = read_json(mf)
        gates = meta["gates"]
        gates_per_image.append(len(gates))
        for g in gates:
            depth.append(g["depth"])
            pitch.append(g["pitch_deg"])
            yaw.append(g["yaw_deg"])
            c = g["center_cam"]
            cx.append(c[0]); cy.append(c[1]); cz.append(c[2])
            pts = np.asarray(g["corners_px"], float)
            x, y = pts[:, 0], pts[:, 1]
            area.append(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
            corner_x.extend(pts[:, 0].tolist())
            corner_y.extend(pts[:, 1].tolist())

    return {
        "depth": np.array(depth), "pitch": np.array(pitch), "yaw": np.array(yaw),
        "cx": np.array(cx), "cy": np.array(cy), "cz": np.array(cz),
        "area": np.array(area), "gates_per_image": np.array(gates_per_image),
        "corner_x": np.array(corner_x), "corner_y": np.array(corner_y),
    }


def _ks_uniform(sample: np.ndarray, lo: float, hi: float) -> float:
    """KS-test p-value of ``sample`` against Uniform(lo, hi). Returns 0 if empty."""
    if sample.size < 8 or hi <= lo:
        return 0.0
    return float(stats.kstest(sample, "uniform", args=(lo, hi - lo)).pvalue)


def _hist_kde(ax, data: np.ndarray, title: str, xlabel: str, color: str) -> None:
    if data.size == 0:
        ax.set_title(f"{title} (no data)")
        return
    ax.hist(data, bins=40, density=True, alpha=0.55, color=color, edgecolor="none")
    if data.size > 8 and np.ptp(data) > 1e-6:
        try:
            kde = stats.gaussian_kde(data)
            xs = np.linspace(data.min(), data.max(), 200)
            ax.plot(xs, kde(xs), color="black", lw=1.5)
        except np.linalg.LinAlgError:
            pass
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")


def plot_distribution(cfg: DotDict | None = None, split: str = "train",
                      out_dir: str | Path | None = None) -> dict:
    """Produce all distribution plots for a split. Returns {plots, stats}."""
    log = get_logger()
    cfg = cfg or load_all()
    style = cfg.report.plots.style
    dpi = int(cfg.report.plots.dpi)
    try:
        plt.style.use(style)
    except OSError:
        pass

    root = dataset_root(cfg.datagen)
    s = collect_stats(root, split)
    out_dir = ensure_dir(out_dir or (root / "plots" / split))
    plots: dict[str, str] = {}

    if s["depth"].size == 0:
        log.warning(f"No labelled gates in split '{split}'; skipping distribution plots.")
        return {"plots": plots, "stats": {"n_gates": 0}}

    # --- 3D scatter of gate centers, colored by yaw ---
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(s["cx"], s["cz"], -s["cy"], c=s["yaw"], cmap="twilight", s=6, alpha=0.6)
    ax.set_xlabel("X right (m)"); ax.set_ylabel("Z depth (m)"); ax.set_zlabel("up (m)")
    ax.set_title(f"Gate centers in camera frame ({split}, n={s['cx'].size})")
    fig.colorbar(p, ax=ax, label="yaw (deg)", shrink=0.6)
    f = out_dir / "centers_3d.png"; fig.savefig(f, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    plots["centers_3d"] = str(f)

    # --- histograms + KDE grid ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    _hist_kde(axes[0, 0], s["depth"], "Depth", "m", "#4C78A8")
    _hist_kde(axes[0, 1], s["pitch"], "Pitch", "deg", "#F58518")
    _hist_kde(axes[0, 2], s["yaw"], "Yaw", "deg", "#54A24B")
    _hist_kde(axes[1, 0], s["cx"], "Center X (cam)", "m", "#E45756")
    _hist_kde(axes[1, 1], s["cy"], "Center Y (cam)", "m", "#72B7B2")
    _hist_kde(axes[1, 2], np.sqrt(s["area"]), "Apparent size (sqrt area)", "px", "#B279A2")
    fig.suptitle(f"Distributions ({split})")
    fig.tight_layout()
    f = out_dir / "histograms.png"; fig.savefig(f, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    plots["histograms"] = str(f)

    # --- corner coverage heatmap ---
    fig, ax = plt.subplots(figsize=(7, 4.2))
    W, H = int(cfg.camera.resolution.width), int(cfg.camera.resolution.height)
    hh = ax.hist2d(s["corner_x"], s["corner_y"], bins=[64, 36], range=[[0, W], [0, H]], cmap="magma")
    ax.set_title(f"Inner-corner coverage ({split})")
    ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)"); ax.invert_yaxis()
    fig.colorbar(hh[3], ax=ax, label="count")
    f = out_dir / "corner_heatmap.png"; fig.savefig(f, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    plots["corner_heatmap"] = str(f)

    # --- gates-per-image bar ---
    fig, ax = plt.subplots(figsize=(5, 3.5))
    vals, counts = np.unique(s["gates_per_image"], return_counts=True)
    ax.bar(vals, counts, color="#4C78A8")
    ax.set_title(f"Labelled gates per image ({split})")
    ax.set_xlabel("gates in image"); ax.set_ylabel("images")
    f = out_dir / "gates_per_image.png"; fig.savefig(f, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    plots["gates_per_image"] = str(f)

    # --- uniformity stats for report indicators ---
    pose = cfg.datagen.pose
    stats_out = {
        "n_gates": int(s["depth"].size),
        "n_images": int(s["gates_per_image"].size),
        "mean_gates_per_image": float(s["gates_per_image"].mean()),
        "depth_range": [float(s["depth"].min()), float(s["depth"].max())],
        "ks_pvalue": {
            "depth": _ks_uniform(s["depth"], float(pose.depth_min), float(pose.depth_max)),
            "pitch": _ks_uniform(s["pitch"], float(pose.pitch_min), float(pose.pitch_max)),
            "yaw": _ks_uniform(s["yaw"], float(pose.yaw_min), float(pose.yaw_max)),
        },
    }
    log.info(f"[{split}] {stats_out['n_gates']} gates, KS p-values: {stats_out['ks_pvalue']}")
    return {"plots": plots, "stats": stats_out}

"""Automatic report generation (Markdown default, optional HTML).

Two entry points:
  * :func:`datagen_report`   -- after data generation (+ distribution plots).
  * :func:`training_report`  -- after training + PnP evaluation.

Each report is a self-contained, timestamped folder under ``reports/`` with all
plots copied into an ``assets/`` subdir and a summary of green/yellow/red
success indicators driven by ``configs/report.toml`` thresholds.
"""

from __future__ import annotations

import base64
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import DotDict, load_all  # noqa: E402
from .utils import ensure_dir, get_logger, timestamp, write_json  # noqa: E402

_GREEN, _YELLOW, _RED = "🟢", "🟡", "🔴"


def _indicator(value, green, yellow, higher_is_better: bool) -> str:
    if value is None:
        return "⚪ n/a"
    if higher_is_better:
        mark = _GREEN if value >= green else (_YELLOW if value >= yellow else _RED)
    else:
        mark = _GREEN if value <= green else (_YELLOW if value <= yellow else _RED)
    return mark


class ReportBuilder:
    """Accumulates markdown sections and copies referenced images into assets/."""

    def __init__(self, report_dir: Path, title: str, fmt: str = "markdown"):
        self.dir = ensure_dir(report_dir)
        self.assets = ensure_dir(report_dir / "assets")
        self.fmt = fmt
        self.lines: list[str] = [f"# {title}", ""]

    def h2(self, text: str) -> None:
        self.lines += [f"## {text}", ""]

    def p(self, text: str) -> None:
        self.lines += [text, ""]

    def table(self, headers: list[str], rows: list[list]) -> None:
        self.lines.append("| " + " | ".join(headers) + " |")
        self.lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in rows:
            self.lines.append("| " + " | ".join(str(c) for c in r) + " |")
        self.lines.append("")

    def image(self, src: str | Path, caption: str = "") -> None:
        src = Path(src)
        if not src.exists():
            self.p(f"_(missing plot: {src.name})_")
            return
        dst = self.assets / src.name
        if src.resolve() != dst.resolve():
            shutil.copy(src, dst)
        if caption:
            self.lines.append(f"**{caption}**")
        self.lines.append(f"![{caption}](assets/{src.name})")
        self.lines.append("")

    def save(self) -> Path:
        md = "\n".join(self.lines)
        if self.fmt == "html":
            out = self.dir / "report.html"
            out.write_text(self._to_html(), encoding="utf-8")
        else:
            out = self.dir / "report.md"
            out.write_text(md, encoding="utf-8")
        return out

    def _to_html(self) -> str:
        # Minimal self-contained HTML: base64-embed the assets.
        html = ["<html><head><meta charset='utf-8'><style>",
                "body{font-family:system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem}",
                "img{max-width:100%;border:1px solid #ddd;border-radius:6px}",
                "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}",
                "</style></head><body>"]
        for line in self.lines:
            if line.startswith("![]") or line.startswith("!["):
                name = line.split("assets/")[-1].rstrip(")")
                p = self.assets / name
                if p.exists():
                    b64 = base64.b64encode(p.read_bytes()).decode()
                    html.append(f"<img src='data:image/png;base64,{b64}'/>")
            elif line.startswith("# "):
                html.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("## "):
                html.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("|"):
                html.append(f"<div>{line}</div>")
            elif line.strip():
                html.append(f"<p>{line}</p>")
        html.append("</body></html>")
        return "\n".join(html)


# ---------------------------------------------------------------------------
# Data-generation report
# ---------------------------------------------------------------------------
def datagen_report(cfg: DotDict, datagen_summary: dict, dist_results: dict) -> Path:
    """Build the post-datagen report. ``dist_results`` maps split -> plot_distribution() output."""
    log = get_logger()
    fmt = cfg.report.output.format
    rdir = Path(cfg.report.output.root) / f"datagen_{cfg.datagen.dataset.name}_{timestamp()}"
    rb = ReportBuilder(rdir, f"Data Generation Report — {cfg.datagen.dataset.name}", fmt)

    th = cfg.report.thresholds
    rb.h2("Summary")
    rb.table(
        ["split", "images", "labelled images", "labelled fraction", "instances"],
        [[s, datagen_summary["images"][s], datagen_summary["labelled_images"][s],
          f"{datagen_summary['labelled_fraction'][s]:.2f} "
          f"{_indicator(datagen_summary['labelled_fraction'][s], th.dataset.min_labelled_fraction, th.dataset.min_labelled_fraction, True)}",
          datagen_summary["instances"][s]] for s in ("train", "val", "test")],
    )
    cam = cfg.camera
    rb.p(f"Camera: {cam.resolution.width}×{cam.resolution.height}, "
         f"fx={cam.intrinsics.fx}, fy={cam.intrinsics.fy}. "
         f"Depth range {cfg.datagen.pose.depth_min}–{cfg.datagen.pose.depth_max} m, "
         f"pitch/yaw ±{cfg.datagen.pose.pitch_max}/±{cfg.datagen.pose.yaw_max}°.")

    for split, res in dist_results.items():
        rb.h2(f"Distribution — {split}")
        st = res.get("stats", {})
        if st.get("n_gates", 0) == 0:
            rb.p("_No labelled gates in this split._")
            continue
        ksp = st.get("ks_pvalue", {})
        rb.table(
            ["metric", "value", "uniformity (KS p)"],
            [["gates", st["n_gates"], ""],
             ["mean gates/image", f"{st['mean_gates_per_image']:.2f}", ""],
             ["depth (deg pitch/yaw sampled uniform)", "", ""],
             ["pitch", "", f"{ksp.get('pitch', 0):.3f} {_indicator(ksp.get('pitch', 0), th.dataset.min_uniformity_pvalue, th.dataset.min_uniformity_pvalue, True)}"],
             ["yaw", "", f"{ksp.get('yaw', 0):.3f} {_indicator(ksp.get('yaw', 0), th.dataset.min_uniformity_pvalue, th.dataset.min_uniformity_pvalue, True)}"]],
        )
        rb.p("_Note: depth appears non-uniform after filtering because close gates are "
             "dropped by the fully-visible constraint — this is expected._")
        for key in ("centers_3d", "histograms", "corner_heatmap", "gates_per_image"):
            if key in res.get("plots", {}):
                rb.image(res["plots"][key])

    out = rb.save()
    write_json(rdir / "datagen_summary.json", datagen_summary)
    log.info(f"Data-gen report -> {out}")
    return out


# ---------------------------------------------------------------------------
# PnP reconstruction plots
# ---------------------------------------------------------------------------
def plot_pnp_reconstruction(cfg: DotDict, pnp_result: dict, out_dir: Path) -> dict:
    """Plot corner-RMSE / pose-error vs checkpoint + a confidence-vs-error scatter."""
    out_dir = ensure_dir(out_dir)
    dpi = int(cfg.report.plots.dpi)
    try:
        plt.style.use(cfg.report.plots.style)
    except OSError:
        pass

    curve = [c for c in pnp_result.get("curve", []) if c.get("mean_trans_err") is not None]
    plots = {}
    if curve:
        ep = [c["epoch"] for c in curve]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        axes[0].plot(ep, [c["mean_corner_rmse"] for c in curve], "o-", color="#4C78A8")
        axes[0].set_title("Corner RMSE vs checkpoint"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("px")
        axes[1].plot(ep, [c["mean_trans_err"] for c in curve], "o-", label="mean", color="#F58518")
        axes[1].plot(ep, [c["wmean_trans_err"] for c in curve], "s--", label="conf-weighted", color="#E45756")
        axes[1].set_title("PnP translation error"); axes[1].set_xlabel("epoch"); axes[1].set_ylabel("m"); axes[1].legend()
        axes[2].plot(ep, [c["mean_rot_err"] for c in curve], "o-", label="mean", color="#54A24B")
        axes[2].plot(ep, [c["wmean_rot_err"] for c in curve], "s--", label="conf-weighted", color="#72B7B2")
        axes[2].set_title("PnP rotation error"); axes[2].set_xlabel("epoch"); axes[2].set_ylabel("deg"); axes[2].legend()
        fig.tight_layout()
        f = out_dir / "pnp_reconstruction.png"; fig.savefig(f, dpi=dpi, bbox_inches="tight"); plt.close(fig)
        plots["pnp_reconstruction"] = str(f)

    # confidence-vs-error scatter for the best/last checkpoint (richest records)
    ckpts = [c for c in pnp_result.get("checkpoints", []) if c.get("records")]
    if ckpts:
        best = max(ckpts, key=lambda c: c["epoch"])
        recs = best["records"]
        conf = np.array([r["mean_conf"] for r in recs])
        terr = np.array([r["trans_err"] for r in recs])
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(conf, terr, s=10, alpha=0.5, color="#B279A2")
        ax.set_xlabel("mean keypoint confidence"); ax.set_ylabel("translation error (m)")
        ax.set_title(f"Confidence vs error ({best['checkpoint']})")
        f = out_dir / "confidence_vs_error.png"; fig.savefig(f, dpi=dpi, bbox_inches="tight"); plt.close(fig)
        plots["confidence_vs_error"] = str(f)

    return {"plots": plots}


# ---------------------------------------------------------------------------
# Training + PnP report
# ---------------------------------------------------------------------------
def training_report(cfg: DotDict, train_result: dict, curve_result: dict,
                    montage_result: dict, pnp_result: dict) -> Path:
    """Build the post-training report combining training curves + PnP reconstruction."""
    log = get_logger()
    from .train import resolve_model
    fmt = cfg.report.output.format
    rdir = Path(cfg.report.output.root) / f"training_{cfg.datagen.dataset.name}_{timestamp()}"
    rb = ReportBuilder(rdir, f"Training + PnP Report — {resolve_model(cfg)['weights']}", fmt)
    th = cfg.report.thresholds

    # --- headline success indicators ---
    final = curve_result.get("final_row", {})
    pose_map = final.get("metrics/mAP50(P)") or final.get("metrics/mAP50(B)")
    best_pnp = None
    curve = [c for c in pnp_result.get("curve", []) if c.get("mean_trans_err") is not None]
    if curve:
        best_pnp = max(curve, key=lambda c: c["epoch"])

    rb.h2("Success indicators")
    rows = [["pose mAP@50", f"{pose_map:.3f}" if pose_map is not None else "n/a",
             _indicator(pose_map, th.detection.pose_map50_green, th.detection.pose_map50_yellow, True)]]
    if best_pnp:
        rows += [
            ["corner RMSE (px)", f"{best_pnp['mean_corner_rmse']:.2f}",
             _indicator(best_pnp["mean_corner_rmse"], th.corners.rmse_px_green, th.corners.rmse_px_yellow, False)],
            ["PnP translation (m)", f"{best_pnp['mean_trans_err']:.3f}",
             _indicator(best_pnp["mean_trans_err"], th.pnp.translation_m_green, th.pnp.translation_m_yellow, False)],
            ["PnP rotation (deg)", f"{best_pnp['mean_rot_err']:.2f}",
             _indicator(best_pnp["mean_rot_err"], th.pnp.rotation_deg_green, th.pnp.rotation_deg_yellow, False)],
        ]
    rb.table(["metric", "value", "status"], rows)
    rb.p(f"Run directory: `{train_result.get('run_dir', '')}`")

    # --- training curves ---
    rb.h2("Training progress")
    for key in ("losses", "metrics"):
        if key in curve_result.get("plots", {}):
            rb.image(curve_result["plots"][key])
    if "overlay_montage" in montage_result.get("plots", {}):
        rb.h2("Predicted (orange) vs ground-truth (green) corners")
        rb.image(montage_result["plots"]["overlay_montage"])

    # --- PnP reconstruction ---
    rb.h2("PnP reconstruction error vs training checkpoint")
    if not curve:
        rb.p("_No PnP matches were produced: the model's detections did not match "
             "any ground-truth gate within the matching tolerance. This is expected "
             "for an undertrained model (see the low pose mAP above) — train longer or "
             "with more data, then re-run `scripts/evaluate_pnp.py`._")
    else:
        recon = plot_pnp_reconstruction(cfg, pnp_result, rdir / "assets")
        for key in ("pnp_reconstruction", "confidence_vs_error"):
            if key in recon.get("plots", {}):
                rb.image(recon["plots"][key])
        rb.table(
            ["checkpoint", "epoch", "matched", "corner RMSE px", "trans err m", "rot err deg"],
            [[c["checkpoint"], c["epoch"], c["n_matched"], f"{c['mean_corner_rmse']:.2f}",
              f"{c['mean_trans_err']:.3f}", f"{c['mean_rot_err']:.2f}"] for c in curve],
        )

    out = rb.save()
    log.info(f"Training report -> {out}")
    return out

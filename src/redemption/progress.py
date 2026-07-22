"""Shared live-progress helpers: results parsing, the dashboard, and the
in-process training callback that grows the live PnP reconstruction curve.

Kept separate from train.py / watch.py to avoid circular imports. The preferred
way to get live progress is the in-process callback (:func:`register_live_progress`),
registered by :func:`redemption.train.train`. It runs the subset PnP evaluation
inside the training process, so it never spawns a second torch process (which on
Windows can trip WinError 1455 "paging file too small" under memory pressure).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import DotDict  # noqa: E402
from .dataset import dataset_root  # noqa: E402
from .pnp import evaluate_checkpoint  # noqa: E402
from .utils import ensure_dir, get_logger, write_json  # noqa: E402


def read_results(run_dir: Path) -> pd.DataFrame | None:
    csv = Path(run_dir) / "results.csv"
    if not csv.exists():
        return None
    try:
        df = pd.read_csv(csv)
    except Exception:  # noqa: BLE001 - may be mid-write
        return None
    if df.empty:
        return None
    df.columns = [c.strip() for c in df.columns]
    return df


def latest_epoch(df: pd.DataFrame) -> int:
    if "epoch" in df.columns:
        return int(df["epoch"].iloc[-1])
    return int(len(df))


def eval_last_checkpoint(run_dir: Path, cfg: DotDict, epoch: int, subset: int,
                         device, safe_copy: bool = True) -> dict | None:
    """PnP-evaluate last.pt on a small test subset. Returns a compact result dict.

    ``safe_copy`` copies the checkpoint first (needed when an external process
    might read it mid-write); the in-process callback can skip that.
    """
    log = get_logger()
    src = Path(run_dir) / "weights" / "last.pt"
    if not src.exists():
        return None
    weights = src
    tmp = None
    try:
        if safe_copy:
            tmp = ensure_dir(Path(run_dir) / "report_plots") / "_live_last.pt"
            shutil.copy(src, tmp)
            weights = tmp
        res = evaluate_checkpoint(weights, cfg, max_images=subset, device=device)
    except Exception as exc:  # noqa: BLE001 - never let one epoch kill progress
        log.warning(f"live PnP eval failed at epoch {epoch}: {exc}")
        return None
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    res["epoch"] = int(epoch)
    res.pop("records", None)
    return res


def plot_dashboard(run_dir: Path, cfg: DotDict, df: pd.DataFrame,
                   curve: list[dict], out_dir: Path,
                   val_curve: list[dict] | None = None) -> Path:
    """One-glance dashboard: losses + metrics + live PnP reconstruction error."""
    dpi = int(cfg.report.plots.dpi)
    try:
        plt.style.use(cfg.report.plots.style)
    except OSError:
        pass

    epochs = df["epoch"] if "epoch" in df.columns else range(len(df))
    pts = [c for c in curve if c.get("mean_trans_err") is not None]
    latest = latest_epoch(df)

    fig, ax = plt.subplots(2, 2, figsize=(15, 9))

    for c in [c for c in df.columns if "loss" in c]:
        ax[0, 0].plot(epochs, df[c], lw=1.3, marker=".", label=c)
    ax[0, 0].set_title("Losses"); ax[0, 0].set_xlabel("epoch"); ax[0, 0].legend(fontsize=6, ncol=2)

    for c in [c for c in df.columns if c.startswith("metrics/")]:
        ax[0, 1].plot(epochs, df[c], lw=1.3, marker=".", label=c.replace("metrics/", ""))
    ax[0, 1].set_title("Detection / pose metrics"); ax[0, 1].set_xlabel("epoch")
    ax[0, 1].legend(fontsize=6, ncol=2)

    vpts = [c for c in (val_curve or []) if c.get("corner_rmse_med") is not None]
    if vpts:
        # REAL-val corner-error convergence (label-based; needs no GT poses).
        vep = [c["epoch"] for c in vpts]
        ax[1, 0].plot(vep, [c["corner_rmse_med"] for c in vpts], "o-", color="#4C78A8", label="median")
        ax[1, 0].plot(vep, [c["corner_rmse_p90"] for c in vpts], "s--", color="#E45756", label="p90")
        ax[1, 0].set_title("REAL val corner RMSE vs epoch"); ax[1, 0].set_xlabel("epoch")
        ax[1, 0].set_ylabel("px"); ax[1, 0].legend(fontsize=7)
        ax[1, 1].plot(vep, [100 * c["det_rate"] for c in vpts], "o-", color="#54A24B")
        ax[1, 1].set_title("REAL val detection rate"); ax[1, 1].set_xlabel("epoch"); ax[1, 1].set_ylabel("%")
        head = f"val corner {vpts[-1]['corner_rmse_med']:.2f}px  det {100 * vpts[-1]['det_rate']:.1f}%"
    elif pts:
        ep = [c["epoch"] for c in pts]
        ax[1, 0].plot(ep, [c["mean_corner_rmse"] for c in pts], "o-", color="#4C78A8")
        ax[1, 0].set_title("PnP corner RMSE (live subset)"); ax[1, 0].set_xlabel("epoch"); ax[1, 0].set_ylabel("px")
        ax[1, 1].plot(ep, [c["mean_trans_err"] for c in pts], "o-", color="#F58518", label="trans (m)")
        ax[1, 1].plot(ep, [c["wmean_trans_err"] for c in pts], "s--", color="#E45756", label="trans w (m)")
        ax[1, 1].set_ylabel("translation err (m)"); ax[1, 1].set_xlabel("epoch")
        axr = ax[1, 1].twinx()
        axr.plot(ep, [c["mean_rot_err"] for c in pts], "^-", color="#54A24B", label="rot (deg)")
        axr.set_ylabel("rotation err (deg)")
        h1, la1 = ax[1, 1].get_legend_handles_labels()
        h2, la2 = axr.get_legend_handles_labels()
        ax[1, 1].legend(h1 + h2, la1 + la2, fontsize=7, loc="upper right")
        ax[1, 1].set_title("PnP reconstruction error (live subset)")
        head = f"live PnP matches: {pts[-1]['n_matched']}"
    else:
        ax[1, 0].set_title("corner RMSE (warming up)")
        ax[1, 1].text(0.5, 0.5, "no eval yet", ha="center", va="center", transform=ax[1, 1].transAxes)
        head = "warming up"

    fig.suptitle(f"Training progress — epoch {latest}  |  {head}", fontsize=13)
    fig.tight_layout()
    out = ensure_dir(out_dir) / "progress_dashboard.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


_VAL_CACHE: dict = {}


def _load_val_subset(cfg: DotDict, n: int):
    """Load a fixed subset of val images + labeled corners into memory (once)."""
    import cv2
    root = dataset_root(cfg.datagen)
    vi, vl = root / "images" / "val", root / "labels" / "val"
    W = int(cfg.camera.resolution.width)
    H = int(cfg.camera.resolution.height)
    out = []
    for ip in sorted(vi.glob("*"))[:n]:
        lp = vl / (ip.stem + ".txt")
        if not lp.exists():
            continue
        img = cv2.imread(str(ip))
        if img is None:
            continue
        gts = []
        for line in lp.read_text().strip().splitlines():
            p = line.split()
            if len(p) < 17:
                continue
            gts.append(np.array([[float(p[5 + i * 3]) * W, float(p[5 + i * 3 + 1]) * H]
                                 for i in range(4)], float))
        if gts:
            out.append((img, gts))
    return out


def eval_val_corners(run_dir: Path, cfg: DotDict, n_images: int, device) -> dict | None:
    """Corner RMSE of predicted vs LABELED corners on a val subset.

    A real-data convergence metric that needs no GT poses (unlike the PnP eval):
    just runs the current checkpoint on a cached val subset and compares to the
    labeled corners. Returns median/p90 px + detection rate.
    """
    from .infer import infer_image, load_model
    from .metrics import corner_rmse, match_by_center
    if "cache" not in _VAL_CACHE:
        _VAL_CACHE["cache"] = _load_val_subset(cfg, n_images)
    cache = _VAL_CACHE["cache"]
    weights = Path(run_dir) / "weights" / "last.pt"
    if not cache or not weights.exists():
        return None
    model = load_model(str(weights))
    imgsz = int(cfg.train.train.imgsz)
    rmses, n_gt, n_match = [], 0, 0
    for img, gts in cache:
        n_gt += len(gts)
        dets = infer_image(model, img, conf=0.25, imgsz=imgsz, device=device)
        if not dets:
            continue
        pc = np.array([d.center_px for d in dets])
        gc = np.array([g.mean(0) for g in gts])
        for pi, gi in match_by_center(pc, gc, 60.0):
            rmses.append(corner_rmse(dets[pi].kpts_px, gts[gi]))
            n_match += 1
    if not rmses:
        return None
    a = np.array(rmses)
    return {"corner_rmse_med": float(np.median(a)),
            "corner_rmse_p90": float(np.percentile(a, 90)),
            "det_rate": n_match / max(n_gt, 1), "n": len(rmses)}


def register_live_progress(model, cfg: DotDict) -> None:
    """Register an Ultralytics callback that refreshes live progress each epoch.

    Runs entirely inside the training process (no second torch process). Fires on
    ``on_model_save`` -- right after ``last.pt`` is written -- so the subset PnP
    eval uses the current epoch's weights. Refreshes
    ``report_plots/progress_dashboard.png`` + ``pnp_progress.json`` every epoch.
    Controlled by ``[live].enabled`` (default True).
    """
    live = cfg.train.live
    if not bool(live.get("enabled", True)):
        return
    subset = int(live.subset_images)
    device = live.get("device", cfg.train.train.device)
    val_eval = bool(live.get("val_corner_eval", False))
    val_n = int(live.get("val_corner_images", 150))
    state: dict = {"curve": [], "val_curve": []}
    log = get_logger()

    def _cb(trainer) -> None:
        try:
            run_dir = Path(trainer.save_dir)
            out = ensure_dir(run_dir / "report_plots")
            epoch = int(getattr(trainer, "epoch", 0)) + 1
            res = eval_last_checkpoint(run_dir, cfg, epoch, subset, device, safe_copy=False)
            if res is not None:
                state["curve"].append(res)
                write_json(out / "pnp_progress.json",
                           {"run_dir": str(run_dir), "curve": state["curve"]})
                log.info(f"  live epoch {epoch}: PnP matches={res['n_matched']} "
                         f"corner_rmse={res.get('mean_corner_rmse')} "
                         f"trans_err={res.get('mean_trans_err')}")
            # REAL-val corner-error convergence (label-based; no GT poses needed)
            if val_eval:
                vc = eval_val_corners(run_dir, cfg, val_n, device)
                if vc is not None:
                    vc["epoch"] = epoch
                    state["val_curve"].append(vc)
                    write_json(out / "val_corner_progress.json", {"curve": state["val_curve"]})
                    log.info(f"  live epoch {epoch}: val corner RMSE "
                             f"{vc['corner_rmse_med']:.2f}px (p90 {vc['corner_rmse_p90']:.2f}), "
                             f"det {100 * vc['det_rate']:.1f}%")
            df = read_results(run_dir)
            if df is not None:
                plot_dashboard(run_dir, cfg, df, state["curve"], out, val_curve=state["val_curve"])
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001 - callbacks must never crash training
            log.warning(f"live progress callback failed: {exc}")

    model.add_callback("on_model_save", _cb)
    log.info("Live in-training progress enabled (dashboard refreshes each epoch).")

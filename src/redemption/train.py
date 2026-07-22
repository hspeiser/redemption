"""YOLO-Pose training wrapper + training-progress plots.

All hyper-parameters come from ``configs/train.toml``. Ultralytics handles
checkpointing natively (``best.pt``/``last.pt`` + periodic ``epochN.pt`` via
``save_period``) and logs every loss/metric to ``results.csv``; we parse that
for the loss/metric curves and render a predicted-vs-GT overlay montage.

The corner-RMSE-vs-checkpoint and PnP-error-vs-checkpoint curves are produced by
:mod:`redemption.pnp` (a single per-checkpoint sweep), not here, to avoid a
duplicate inference pass.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import DotDict, load_all  # noqa: E402
from .dataset import dataset_root, split_paths, write_data_yaml  # noqa: E402
from .infer import infer_image, load_model  # noqa: E402
from .progress import register_live_progress  # noqa: E402
from .utils import ensure_dir, get_logger, read_json  # noqa: E402
from .viz import montage, overlay_pred_gt  # noqa: E402


def build_overrides(cfg: DotDict, data_yaml: Path) -> dict:
    t = cfg.train.train
    ck = cfg.train.checkpoints
    aug = cfg.train.augment
    return {
        "data": str(data_yaml),
        "epochs": int(t.epochs),
        "batch": int(t.batch),
        "imgsz": int(t.imgsz),
        "device": t.device,
        "workers": int(t.workers),
        "amp": bool(t.amp),
        "patience": int(t.patience),
        "seed": int(t.seed),
        "cos_lr": bool(t.cos_lr),
        "project": str(t.project),
        "name": str(t.name),
        "exist_ok": bool(t.exist_ok),
        "save": bool(ck.save),
        "save_period": int(ck.save_period),
        "plots": True,
        # augmentation
        "hsv_h": float(aug.hsv_h), "hsv_s": float(aug.hsv_s), "hsv_v": float(aug.hsv_v),
        "degrees": float(aug.degrees), "translate": float(aug.translate),
        "scale": float(aug.scale), "fliplr": float(aug.fliplr),
        "mosaic": float(aug.mosaic), "mixup": float(aug.mixup),
    }


def train(cfg: DotDict | None = None) -> dict:
    """Train a YOLO-Pose model. Returns dict(run_dir, results_csv, metrics)."""
    log = get_logger()
    cfg = cfg or load_all()

    root = dataset_root(cfg.datagen)
    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        write_data_yaml(root)

    model = load_model(str(cfg.train.model.weights))
    register_live_progress(model, cfg)  # live in-process dashboard + PnP curve each epoch
    overrides = build_overrides(cfg, data_yaml)
    log.info(f"Training {cfg.train.model.weights} on {data_yaml} "
             f"(epochs={overrides['epochs']}, batch={overrides['batch']}, device={overrides['device']})")

    results = model.train(**overrides)
    run_dir = Path(model.trainer.save_dir)
    log.info(f"Training complete -> {run_dir}")

    metrics = {}
    try:
        metrics = {k: float(v) for k, v in results.results_dict.items()
                   if isinstance(v, (int, float))}
    except Exception:  # noqa: BLE001 - metrics are best-effort
        pass

    return {"run_dir": str(run_dir), "results_csv": str(run_dir / "results.csv"), "metrics": metrics}


# ---------------------------------------------------------------------------
# Progress plots
# ---------------------------------------------------------------------------
def plot_training_curves(run_dir: Path, cfg: DotDict, out_dir: Path | None = None) -> dict:
    """Parse ``results.csv`` and plot loss + mAP curves. Returns {plots}."""
    log = get_logger()
    csv = Path(run_dir) / "results.csv"
    if not csv.exists():
        log.warning(f"No results.csv at {csv}")
        return {"plots": {}}

    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    epochs = df["epoch"] if "epoch" in df.columns else np.arange(len(df))
    out_dir = ensure_dir(out_dir or (Path(run_dir) / "report_plots"))
    dpi = int(cfg.report.plots.dpi)
    try:
        plt.style.use(cfg.report.plots.style)
    except OSError:
        pass
    plots = {}

    # losses
    loss_cols = [c for c in df.columns if "loss" in c]
    if loss_cols:
        fig, ax = plt.subplots(figsize=(9, 5))
        for c in loss_cols:
            ax.plot(epochs, df[c], label=c, lw=1.4)
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Training / val losses")
        ax.legend(fontsize=7, ncol=2)
        f = out_dir / "losses.png"; fig.savefig(f, dpi=dpi, bbox_inches="tight"); plt.close(fig)
        plots["losses"] = str(f)

    # metrics (mAP / precision / recall)
    metric_cols = [c for c in df.columns if c.startswith("metrics/")]
    if metric_cols:
        fig, ax = plt.subplots(figsize=(9, 5))
        for c in metric_cols:
            ax.plot(epochs, df[c], label=c.replace("metrics/", ""), lw=1.4)
        ax.set_xlabel("epoch"); ax.set_ylabel("value"); ax.set_title("Detection / pose metrics")
        ax.legend(fontsize=7, ncol=2)
        f = out_dir / "metrics.png"; fig.savefig(f, dpi=dpi, bbox_inches="tight"); plt.close(fig)
        plots["metrics"] = str(f)

    return {"plots": plots, "final_row": df.iloc[-1].to_dict() if len(df) else {}}


def overlay_montage(cfg: DotDict, run_dir: Path, out_dir: Path | None = None) -> dict:
    """Run best.pt on a few test images and save a predicted-vs-GT montage."""
    log = get_logger()
    weights = Path(run_dir) / "weights" / "best.pt"
    if not weights.exists():
        weights = Path(run_dir) / "weights" / "last.pt"
    if not weights.exists():
        log.warning("No checkpoint found for overlay montage.")
        return {"plots": {}}

    root = dataset_root(cfg.datagen)
    paths = split_paths(root, "test")
    meta_files = sorted(paths["meta"].glob("*.json"))[: int(cfg.train.progress.overlay_samples)]
    if not meta_files:
        return {"plots": {}}

    model = load_model(str(weights))
    tiles = []
    for mf in meta_files:
        meta = read_json(mf)
        img = cv2.imread(str(paths["images"] / meta["image"]))
        dets = infer_image(model, img, conf=0.25, imgsz=int(cfg.train.train.imgsz),
                           device=cfg.train.train.device)
        preds = [d.kpts_px for d in dets]
        gts = [np.asarray(g["corners_px"], float) for g in meta["gates"]]
        tiles.append(overlay_pred_gt(img, preds, gts))

    out_dir = ensure_dir(out_dir or (Path(run_dir) / "report_plots"))
    grid = montage(tiles, cols=4)
    f = out_dir / "overlay_montage.png"
    cv2.imwrite(str(f), grid)
    return {"plots": {"overlay_montage": str(f)}}

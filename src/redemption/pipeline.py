"""High-level pipeline stages that stitch the modules together.

These are the functions the ``scripts/*.py`` entrypoints call. Each stage is
config-driven and auto-generates its report.
"""

from __future__ import annotations

from pathlib import Path

from .config import DotDict, load_all
from .datagen import generate
from .dataset import SPLITS
from .distribution_plots import plot_distribution
from .pnp import evaluate_run
from .report import datagen_report, training_report
from .train import overlay_montage, plot_training_curves, train
from .utils import get_logger


def run_datagen(cfg: DotDict | None = None) -> dict:
    """Generate data + distribution plots + data-gen report."""
    cfg = cfg or load_all()
    summary = generate(cfg)
    dist = {s: plot_distribution(cfg, s) for s in SPLITS}
    report = datagen_report(cfg, summary, dist)
    return {"summary": summary, "dist": dist, "report": report}


def run_distribution(cfg: DotDict | None = None) -> dict:
    """Just (re)build distribution plots for every split."""
    cfg = cfg or load_all()
    return {s: plot_distribution(cfg, s) for s in SPLITS}


def run_training(cfg: DotDict | None = None) -> dict:
    """Train + training-progress plots + PnP evaluation + training report."""
    cfg = cfg or load_all()
    tr = train(cfg)
    run_dir = Path(tr["run_dir"])

    curves = plot_training_curves(run_dir, cfg)
    mont = overlay_montage(cfg, run_dir)

    # Point PnP eval at the run we just trained.
    cfg.pnp.model.run_dir = str(run_dir)
    pnp_res = evaluate_run(cfg)

    report = training_report(cfg, tr, curves, mont, pnp_res)
    return {"train": tr, "curves": curves, "montage": mont, "pnp": pnp_res, "report": report}


def run_pnp_eval(cfg: DotDict | None = None) -> dict:
    """Run PnP evaluation over an existing run + rebuild the training report."""
    cfg = cfg or load_all()
    pnp_res = evaluate_run(cfg)
    run_dir = Path(pnp_res["run_dir"])
    curves = plot_training_curves(run_dir, cfg)
    mont = overlay_montage(cfg, run_dir)
    report = training_report(cfg, {"run_dir": str(run_dir)}, curves, mont, pnp_res)
    return {"pnp": pnp_res, "report": report}


def run_full(cfg: DotDict | None = None) -> dict:
    """Full end-to-end pipeline: datagen -> distribution -> train -> PnP -> reports."""
    log = get_logger()
    cfg = cfg or load_all()
    log.info("=== STAGE 1/2: data generation ===")
    dg = run_datagen(cfg)
    log.info("=== STAGE 2/2: training + PnP ===")
    tr = run_training(cfg)
    log.info(f"Done. Reports:\n  datagen : {dg['report']}\n  training: {tr['report']}")
    return {"datagen": dg, "training": tr}

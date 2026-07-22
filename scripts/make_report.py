"""Rebuild reports from existing artifacts (no data-gen, no training).

Rebuilds the distribution/data-gen report from the current dataset, and the
training+PnP report from the configured (or latest) run.
Run:  uv run python scripts/make_report.py
"""

from redemption.config import load_all
from redemption.datagen import dataset_root
from redemption.distribution_plots import plot_distribution
from redemption.dataset import SPLITS
from redemption.report import datagen_report
from redemption.pipeline import run_pnp_eval
from redemption.utils import get_logger, read_json


def main() -> None:
    log = get_logger()
    cfg = load_all()

    root = dataset_root(cfg.datagen)
    summary_path = root / "datagen_summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        dist = {s: plot_distribution(cfg, s) for s in SPLITS}
        path = datagen_report(cfg, summary, dist)
        print(f"Data-gen report: {path}")
    else:
        log.warning(f"No datagen_summary.json at {root}; skipping data-gen report.")

    try:
        result = run_pnp_eval(cfg)
        print(f"Training/PnP report: {result['report']}")
    except FileNotFoundError as exc:
        log.warning(f"Skipping training report: {exc}")


if __name__ == "__main__":
    main()

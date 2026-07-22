"""Run PnP evaluation over an existing training run and rebuild the report.

Useful to re-evaluate without retraining. Config-only: edit configs/pnp.toml
(set run_dir, solver, confidence handling). Empty run_dir => latest run.
Run:  uv run python scripts/evaluate_pnp.py
"""

from redemption.pipeline import run_pnp_eval


def main() -> None:
    result = run_pnp_eval()
    print(f"\nPnP/training report: {result['report']}")


if __name__ == "__main__":
    main()

"""Generate the synthetic dataset, distribution plots, and a data-gen report.

Config-only: edit configs/datagen.toml (and camera/gate). No CLI args.
Run:  uv run python scripts/generate_data.py
"""

from redemption.pipeline import run_datagen


def main() -> None:
    result = run_datagen()
    print(f"\nData-gen report: {result['report']}")


if __name__ == "__main__":
    main()

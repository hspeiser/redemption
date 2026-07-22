"""Train the YOLO-Pose model, then auto-run PnP eval and build the report.

Config-only: edit configs/train.toml (model size, epochs, batch, device...).
Requires a dataset generated first (scripts/generate_data.py).
Run:  uv run python scripts/train_model.py
"""

from redemption.pipeline import run_training


def main() -> None:
    result = run_training()
    print(f"\nTraining report: {result['report']}")


if __name__ == "__main__":
    main()

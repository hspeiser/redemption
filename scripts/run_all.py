"""Full end-to-end pipeline: datagen -> distribution -> train -> PnP -> reports.

This is the one-command entrypoint. Everything is driven by configs/*.toml.
For a fast smoke test, shrink the counts in configs/datagen.toml and epochs in
configs/train.toml (see README).
Run:  uv run python scripts/run_all.py
"""

from redemption.pipeline import run_full


def main() -> None:
    result = run_full()
    print("\n=== Reports ===")
    print(f"datagen : {result['datagen']['report']}")
    print(f"training: {result['training']['report']}")


if __name__ == "__main__":
    main()

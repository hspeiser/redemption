"""Convenience entrypoint. The real work lives in scripts/ and src/redemption/.

    uv run python main.py            # full pipeline (== scripts/run_all.py)

Everything is configured via configs/*.toml -- there are no command-line args.
See README.md for the staged workflow and individual stage scripts.
"""

from redemption.pipeline import run_full


def main() -> None:
    run_full()


if __name__ == "__main__":
    main()

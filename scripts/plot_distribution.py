"""(Re)build dataset distribution plots for every split.

Config-only. Run:  uv run python scripts/plot_distribution.py
"""

from redemption.pipeline import run_distribution


def main() -> None:
    results = run_distribution()
    for split, res in results.items():
        for name, path in res.get("plots", {}).items():
            print(f"[{split}] {name}: {path}")


if __name__ == "__main__":
    main()

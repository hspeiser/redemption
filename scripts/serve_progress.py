"""Serve a live training-progress website (auto-refreshing dashboard).

Reads the latest run under runs/ and serves it over HTTP so you can watch
progress in a browser instead of SSH-ing in. Over a Tailscale tailnet, visit
http://<machine-tailnet-ip>:<port>/ .

Config-only: see the [web] section of configs/report.toml. No CLI args.
Run:  uv run python scripts/serve_progress.py
"""

from redemption.web import serve


def main() -> None:
    serve()


if __name__ == "__main__":
    main()

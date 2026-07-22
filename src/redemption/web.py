"""Tiny stdlib HTTP server that shows live training progress in a browser.

Serves an auto-refreshing page with the latest run's progress dashboard
(losses, metrics, corner RMSE, PnP reconstruction error) plus a compact metrics
table read from ``results.csv`` and the live PnP curve. No external deps, so it
runs anywhere the package is installed.

Point a browser at ``http://<host>:<port>/`` -- over a Tailscale tailnet that is
just the machine's tailnet IP, so you never have to SSH in to check progress.

Config: the ``[web]`` section of ``configs/report.toml``. Run via
``scripts/serve_progress.py``.
"""

from __future__ import annotations

import csv
import html
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import DotDict, load_all
from .pnp import latest_run
from .utils import get_logger, read_json


def _run_dir(cfg: DotDict) -> Path | None:
    return latest_run(Path(cfg.train.train.project), cfg.train.train.name)


def _last_csv_row(run_dir: Path) -> dict | None:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except Exception:  # noqa: BLE001
        return None
    if not rows:
        return None
    return {k.strip(): v for k, v in rows[-1].items()}


def _pnp_latest(run_dir: Path) -> dict | None:
    p = run_dir / "report_plots" / "pnp_progress.json"
    if not p.exists():
        return None
    try:
        curve = read_json(p).get("curve", [])
    except Exception:  # noqa: BLE001
        return None
    return curve[-1] if curve else None


def _page(cfg: DotDict) -> bytes:
    refresh = int(cfg.report.web.refresh_seconds)
    run = _run_dir(cfg)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<meta http-equiv='refresh' content='{refresh}'>",
        "<title>redemption — training progress</title>",
        "<style>body{font-family:system-ui,Segoe UI,sans-serif;background:#0f1117;color:#e6e6e6;"
        "margin:0;padding:24px;max-width:1200px;margin:auto}h1{font-weight:600}"
        "img{max-width:100%;border:1px solid #2a2d37;border-radius:8px;background:#fff}"
        "table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #2a2d37;padding:4px 10px;"
        "font-size:14px;text-align:left}.muted{color:#8b8f9a}.big{font-size:22px;font-weight:600}"
        "code{background:#1a1d26;padding:2px 6px;border-radius:4px}</style></head><body>",
        "<h1>🟠 redemption — live training progress</h1>",
    ]

    if run is None:
        parts.append("<p class='muted'>Waiting for a training run to start…</p>")
        parts.append(f"<p class='muted'>auto-refreshing every {refresh}s</p></body></html>")
        return "".join(parts).encode()

    row = _last_csv_row(run) or {}
    pnp = _pnp_latest(run)
    epoch = row.get("epoch", "—")
    parts.append(f"<p class='big'>Epoch {html.escape(str(epoch))}</p>")
    parts.append(f"<p class='muted'>run: <code>{html.escape(str(run))}</code></p>")

    # metrics table
    keys = [k for k in ("metrics/mAP50(P)", "metrics/mAP50(B)", "train/pose_loss",
                        "train/box_loss", "val/pose_loss") if k in row]
    if keys:
        parts.append("<table><tr>" + "".join(f"<th>{html.escape(k)}</th>" for k in keys) + "</tr><tr>"
                     + "".join(f"<td>{html.escape(str(row[k]))}</td>" for k in keys) + "</tr></table>")

    # live PnP
    if pnp and pnp.get("mean_trans_err") is not None:
        parts.append(
            "<table><tr><th>PnP matches</th><th>corner RMSE (px)</th><th>trans err (m)</th>"
            "<th>rot err (deg)</th></tr><tr>"
            f"<td>{pnp.get('n_matched')}</td><td>{pnp.get('mean_corner_rmse'):.2f}</td>"
            f"<td>{pnp.get('mean_trans_err'):.3f}</td><td>{pnp.get('mean_rot_err'):.2f}</td></tr></table>")

    dash = run / "report_plots" / "progress_dashboard.png"
    if dash.exists():
        parts.append(f"<img src='/dashboard.png?t={int(dash.stat().st_mtime)}'/>")
    else:
        parts.append("<p class='muted'>Dashboard not generated yet (waiting for first epoch)…</p>")

    parts.append(f"<p class='muted'>auto-refreshing every {refresh}s — {time.strftime('%X')}</p>")
    parts.append("</body></html>")
    return "".join(parts).encode()


def _make_handler(cfg: DotDict):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence per-request logging
            pass

        def do_GET(self):
            if self.path.startswith("/dashboard.png"):
                run = _run_dir(cfg)
                dash = run / "report_plots" / "progress_dashboard.png" if run else None
                if dash and dash.exists():
                    data = dash.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_error(404)
                return
            body = _page(cfg)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(cfg: DotDict | None = None) -> None:
    log = get_logger()
    cfg = cfg or load_all()
    web = cfg.report.web
    host, port = str(web.host), int(web.port)
    httpd = ThreadingHTTPServer((host, port), _make_handler(cfg))
    log.info(f"Progress site on http://{host}:{port}/  (refresh {web.refresh_seconds}s). Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("progress site stopped.")

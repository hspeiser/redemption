"""Serve the persistent VQ2 flywheel dashboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aigp.vq2_pipeline_dashboard import PipelineDashboardServer, PipelineScanner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8900)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\ai-gp"))
    parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args()
    scanner = PipelineScanner(
        repo=args.repo,
        data_root=args.data_root,
        training_root=args.data_root / "training",
        worldmodel_root=args.data_root / "worldmodel",
        manifest_root=args.data_root / "corpus_manifests",
    )
    print(f"VQ2 pipeline dashboard: http://{args.host}:{args.port}/", flush=True)
    PipelineDashboardServer(scanner, host=args.host, port=args.port).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build a portable, deduplicated bundle for offline/live calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ASSET_KEYS = (
    "ppo_residual_schedule",
    "ppo_residual_checkpoint",
    "seed_checkpoint",
    "demo",
    "map",
    "gate_primary",
    "crop",
    "line_model",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--remote-prefix",
        default="worldmodel/calibration_bundle_v1",
        help="Path to the bundle relative to the remote repository cwd.",
    )
    args = parser.parse_args()

    assets = args.out / "assets"
    sessions = args.out / "g0g4_bundle"
    assets.mkdir(parents=True, exist_ok=True)
    sessions.mkdir(parents=True, exist_ok=True)
    manifest = {"sessions": [], "assets": {}}
    # Campaigns have used both g0g4_* and vq2_g0g4_* root names.  Restrict by
    # path component rather than a root-only glob so newer live calibration
    # sessions are not silently omitted.
    logs = sorted(
        path for path in args.training_root.glob("**/episodes.jsonl")
        if any("g0g4" in part.lower() for part in path.parts)
    )
    for index, log in enumerate(logs):
        config_path = log.parent / "config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text())
        runtime_args = config.get("args", config)
        rewritten = json.loads(json.dumps(config))
        rewritten_args = rewritten.get("args", rewritten)
        session_dir = sessions / f"{index:04d}_{log.parent.name}"
        session_dir.mkdir(parents=True, exist_ok=True)
        for key in ASSET_KEYS:
            value = runtime_args.get(key)
            if value in {None, "", "None"}:
                continue
            source = Path(value)
            if not source.is_absolute():
                source = Path.cwd() / source
            if not source.is_file():
                continue
            sha = digest(source)
            target_name = f"{sha}{source.suffix.lower()}"
            target = assets / target_name
            if not target.exists():
                shutil.copy2(source, target)
            rewritten_args[key] = (
                f"{args.remote_prefix}/assets/{target_name}"
            )
            manifest["assets"].setdefault(sha, {
                "file": target_name,
                "bytes": source.stat().st_size,
                "source": str(source),
            })
        (session_dir / "config.json").write_text(
            json.dumps(rewritten, indent=2) + "\n"
        )
        shutil.copy2(log, session_dir / "episodes.jsonl")
        manifest["sessions"].append({
            "source": str(log.parent),
            "bundle": str(session_dir.relative_to(args.out)),
            "config_sha256": digest(config_path),
            "episodes_sha256": digest(log),
        })
    manifest["session_count"] = len(manifest["sessions"])
    manifest["asset_count"] = len(manifest["assets"])
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({
        "out": str(args.out.resolve()),
        "sessions": manifest["session_count"],
        "assets": manifest["asset_count"],
        "manifest_sha256": digest(args.out / "manifest.json"),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

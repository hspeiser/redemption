"""Launch the live trainer with every campaign argument reproduced exactly.

This avoids argparse-default drift: a saved config is authoritative and only
explicit ``--override key=value`` entries change it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_value(text: str):
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--poc-stop-after-gate", type=int, default=-1)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())["args"]
    for item in args.override:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        if key not in config:
            print(
                f"adding post-config option {key!r}; trainer argparse will "
                "remain authoritative",
                flush=True,
            )
        config[key] = parse_value(value)
    config.update({
        "episodes": args.episodes,
        "output_root": str(args.output_root),
        "eval_only": args.eval_only,
        "poc_stop_after_gate": args.poc_stop_after_gate,
    })
    # Runtime-only and mutually exclusive options are intentionally omitted.
    skip = {"smoke", "replay_dir"}
    command = [sys.executable, str(Path(__file__).with_name(
        "train_vq2_sac_live.py"
    ))]
    for key, value in config.items():
        if key in skip or value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        command.extend([flag, str(value)])
    print(subprocess.list2cmdline(command), flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

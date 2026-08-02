"""Create immutable session-level splits for the VQ2 live-data corpus.

Existing assignments are never changed.  On the first lock, sessions are
deterministically divided into train/validation/policy-selection/final-test.
Subsequent invocations put newly discovered sessions into train by default so
the frozen evaluation pools cannot silently grow after policies have been
selected against them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


STAMP = re.compile(r"(20\d{6}_\d{6})")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_key(entry: dict) -> str:
    host = str(entry.get("host", "unknown")).lower()
    candidates = (
        entry.get("raw_archive_path"), entry.get("training_run_dir"),
        entry.get("session_id"), entry.get("path"),
    )
    for candidate in candidates:
        if not candidate:
            continue
        match = STAMP.search(str(candidate))
        if match:
            return f"{host}:{match.group(1)}"
    return f"{host}:{entry.get('kind')}:{entry.get('session_id', entry.get('path'))}"


def initial_split(key: str) -> str:
    value = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) / 16**16
    if value < 0.05:
        return "final_test"
    if value < 0.15:
        return "policy_selection"
    if value < 0.25:
        return "validation"
    return "train"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--existing", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force-train-token", action="append", default=[])
    parser.add_argument(
        "--assign-new-by-hash", action="store_true",
        help="Allow new sessions to expand frozen evaluation pools.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    previous = {}
    generation = 1
    if args.existing and args.existing.exists():
        old = json.loads(args.existing.read_text(encoding="utf-8"))
        previous = {
            str(row["group_key"]): str(row["split"])
            for row in old.get("groups", [])
        }
        generation = int(old.get("generation", 1)) + 1

    grouped: dict[str, list[dict]] = {}
    for entry in manifest.get("entries", []):
        grouped.setdefault(group_key(entry), []).append(entry)

    rows = []
    for key, entries in sorted(grouped.items()):
        forced = any(token in key for token in args.force_train_token)
        if forced:
            split = "train"
            assignment = "forced_train"
        elif key in previous:
            split = previous[key]
            assignment = "preserved"
        elif previous and not args.assign_new_by_hash:
            split = "train"
            assignment = "new_train_only"
        else:
            split = initial_split(key)
            assignment = "initial_hash"
        rows.append({
            "group_key": key,
            "split": split,
            "assignment": assignment,
            "entry_count": len(entries),
            "kinds": sorted({str(entry.get("kind")) for entry in entries}),
            "paths": sorted({str(entry.get("path")) for entry in entries}),
        })

    counts = Counter(row["split"] for row in rows)
    payload = {
        "schema": 1,
        "generation": generation,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256(args.manifest),
        "policy": (
            "whole timestamp-linked sessions; existing assignments immutable; "
            "new sessions train-only unless explicitly enabled"
        ),
        "split_counts": dict(sorted(counts.items())),
        "groups": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out.resolve()),
        "generation": generation,
        "groups": len(rows),
        "split_counts": payload["split_counts"],
        "sha256": sha256(args.out),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

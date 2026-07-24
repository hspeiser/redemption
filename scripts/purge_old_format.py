"""Delete pre-timestamp-fix episodes (per Henry's request 2026-07-23).

An episode is purged only if ALL of:
  - name starts with rc_20260723_ (July 5 folders are untouched)
  - its mav.jsonl ODOMETRY rows lack time_usec (old recorder format)
  - not modified in the last 5 minutes (not the live recording)
"""

import json
import shutil
import time
from pathlib import Path

ROOT = Path(r"C:\Users\henry\Downloads\AI-GP Simulator v1.0.3379\ai-grand-prix\outputs\captures")


def is_old_format(ep):
    mj = ep / "mav.jsonl"
    if not mj.exists():
        return None  # unknown; don't touch
    checked = 0
    with open(mj) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("msg_id") == 331:
                checked += 1
                if "time_usec" in r:
                    return False
                if checked >= 5:
                    return True
    return None if checked == 0 else True


now = time.time()
freed = 0
for ep in sorted(ROOT.iterdir()):
    if not ep.is_dir() or not ep.name.startswith("rc_20260723_"):
        continue
    newest = max((f.stat().st_mtime for f in ep.glob("*.jsonl")), default=0)
    if now - newest < 300:
        print(f"{ep.name}: SKIP (active)", flush=True)
        continue
    old = is_old_format(ep)
    if old is None:
        print(f"{ep.name}: SKIP (cannot determine format)", flush=True)
        continue
    if not old:
        print(f"{ep.name}: KEEP (native time_usec)", flush=True)
        continue
    size = sum(f.stat().st_size for f in ep.rglob("*") if f.is_file())
    shutil.rmtree(ep)
    freed += size
    print(f"{ep.name}: DELETED ({size/1e9:.2f} GB)", flush=True)

print(f"\nTOTAL freed: {freed/1e9:.1f} GB", flush=True)

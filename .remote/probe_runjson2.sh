#!/bin/bash
python3 - <<'EOF'
import json
from pathlib import Path
base = Path("/mnt/c/Users/henry/lrspeiser/ai-grand-prix/telemetry/plan_sysid")
d = json.load(open(base / "course30_0/run.json"))
print("maneuver:", d["maneuver"])
print("imu_cols:", d["imu_cols"])
print("cmd_cols:", d["cmd_cols"])
print("servo_cols:", d["servo_cols"])
print("frame_cols:", d["frame_cols"])
print("imu row0:", d.get("imu", [[]])[:1])
print("cmd row0:", d.get("cmd", [[]])[:1])
print("keys:", list(d.keys()))
# maneuver types across a sample of runs
from collections import Counter
c = Counter()
dirs = sorted(base.glob("course*"))[:80]
for run_dir in dirs:
    try:
        head = (run_dir / "run.json").open().read(600)
        m = head.split('"maneuver"')[1].split('"')[1] if '"maneuver"' in head else "?"
        c[m] += 1
    except Exception:
        pass
print("maneuver types (first 80 runs):", dict(c))
EOF

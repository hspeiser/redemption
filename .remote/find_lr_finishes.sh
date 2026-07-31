#!/bin/bash
python3 - <<'EOF'
import json
from pathlib import Path

base = Path("/mnt/c/Users/henry/lrspeiser/ai-grand-prix/telemetry/run_outcomes")
hits = []
for f in sorted(base.glob("*.json")):
    try:
        d = json.load(f.open())
    except Exception:
        continue
    text = json.dumps(d).lower()
    blob = {
        "run": d.get("run", f.stem),
        "status": d.get("status"),
    }
    # look for finish/gates indicators anywhere in the doc
    result = d.get("result") or d.get("outcome") or {}
    if isinstance(result, dict):
        for key in ("gates_passed", "max_gate", "finished", "lap_time_s",
                    "final_gate", "official_time_s", "gates"):
            if key in result:
                blob[key] = result[key]
    interesting = (
        '"finished": true' in text
        or "full course" in text or "all 17" in text or "final gate" in text
        or "lap_time" in text or "official" in text and "finish" in text
    )
    if interesting:
        hits.append(blob)
print(f"scanned; candidates: {len(hits)}")
for h in hits[-25:]:
    print(json.dumps(h))
EOF

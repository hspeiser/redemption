#!/bin/bash
BASE=/mnt/c/Users/henry/lrspeiser/ai-grand-prix/outputs
python3 - <<'EOF'
import json
base = "/mnt/c/Users/henry/lrspeiser/ai-grand-prix/outputs"
for name in ("win1", "win2", "win3"):
    try:
        rows = [json.loads(l) for l in open(f"{base}/{name}_loc.jsonl")]
        frames = [r.get("frame", "") for r in rows if r.get("frame")]
        ts = []
        for fr in frames:
            try:
                ts.append(float(fr.replace("l", "").replace(".jpg", "")))
            except ValueError:
                pass
        if ts:
            print(f"{name}: {len(rows)} rows, t {min(ts):.1f} -> {max(ts):.1f}"
                  f" = {max(ts)-min(ts):.1f}s")
    except FileNotFoundError:
        print(name, "missing")
d = json.load(open(f"{base}/_v22_fullcourse_baseline_results_20260715.json"))
print(json.dumps(d, indent=1)[:1200])
EOF

#!/bin/bash
cd ~/aigp
.venv/bin/python - <<'EOF'
import json
import shutil
from pathlib import Path

import numpy as np

root = Path("/mnt/c/Users/henry/aigp_raw/raw_sessions/vq2_20260730_200627")
picks = []
for line in (root / "localizer_debug/debug.jsonl").open():
    try:
        r = json.loads(line)
    except Exception:
        continue
    d = r["debug"]
    exp = [e for e in d.get("expected", []) if e["gate"] == d.get("active_gate")]
    if d.get("fused", 0) == 0 and len(exp) >= 8:
        picks.append((r["image"], d.get("active_gate")))
idx = np.linspace(0, len(picks) - 1, 4).astype(int)
for k, i in enumerate(idx):
    shutil.copy(root / picks[i][0],
                f"/mnt/c/Users/henry/blind_{k}_g{picks[i][1]}.jpg")
    print(picks[i])
EOF

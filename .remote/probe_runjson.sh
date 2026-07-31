#!/bin/bash
F=/mnt/c/Users/henry/lrspeiser/ai-grand-prix/telemetry/plan_sysid/course30_0/run.json
ls -la "$F"
python3 - <<'EOF'
import json
d = json.load(open("/mnt/c/Users/henry/lrspeiser/ai-grand-prix/telemetry/plan_sysid/course30_0/run.json"))
def shape(x, depth=0):
    if isinstance(x, dict):
        return {k: shape(v, depth+1) for k, v in list(x.items())[:12]} if depth < 2 else f"dict({len(x)})"
    if isinstance(x, list):
        return f"list({len(x)}): " + str(shape(x[0], depth+1) if x else "empty")
    return type(x).__name__
print(json.dumps(shape(d), indent=1, default=str)[:2500])
EOF

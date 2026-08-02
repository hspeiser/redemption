"""Diff train_vq2_sac_live.py argparse defaults vs v77's recorded config."""
import json
import re
from pathlib import Path

v77 = json.load(open(
    r"D:\ai-gp\training\vq2_sac_runs\gate10_left15_gpu10_awr_v77"
    r"\20260730_200627\config.json"))["args"]

src = Path(r"scripts\train_vq2_sac_live.py").read_text()
# crude argparse default extraction
pattern = re.compile(
    r"add_argument\(\s*[\"']--([a-z0-9-]+)[\"'][^)]*?default=([^,)]+)",
    re.S)
defaults = {}
for m in pattern.finditer(src):
    key = m.group(1).replace("-", "_")
    val = m.group(2).strip()
    defaults[key] = val

mismatches = []
for key, v77_val in v77.items():
    if key not in defaults:
        continue
    d = defaults[key]
    try:
        d_eval = eval(d, {"__builtins__": {}}, {})
    except Exception:
        d_eval = d
    if isinstance(d_eval, str) and isinstance(v77_val, str):
        same = d_eval == v77_val
    else:
        try:
            same = float(d_eval) == float(v77_val)
        except (TypeError, ValueError):
            same = str(d_eval) == str(v77_val)
    if not same:
        mismatches.append((key, d_eval, v77_val))

print(f"mismatched keys (default vs v77): {len(mismatches)}")
for k, d, v in mismatches:
    print(f"  {k}: default={d!r}  v77={v!r}")

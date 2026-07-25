"""Run vq2_align --label-dump over a list of episodes (one process)."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

lst = Path(sys.argv[1])
out_dir = REPO / "data" / "labels_vq2t"
out_dir.mkdir(parents=True, exist_ok=True)
for ep in [ln.strip() for ln in lst.read_text().splitlines() if ln.strip()]:
    name = Path(ep).name
    out_f = out_dir / f"{name}.npz"
    if out_f.exists():
        print(f"{name}: exists, skip", flush=True)
        continue
    r = subprocess.run(
        [PY, str(REPO / "scripts" / "vq2_align.py"),
         "--episode-dir", ep,
         "--ckpt", str(REPO / "data/models/gatenet_v7_best.pt"),
         "--no-mirror", "--anchor-yaw", "88.5", "--decouple-yaw",
         "--yaw-flip", "--label-dump", str(out_f)],
        capture_output=True, text=True, timeout=1800)
    tail = [ln for ln in r.stdout.splitlines() if "temporal labels" in ln]
    print(f"{name}: {tail[-1] if tail else 'NO LABELS (rc=%d)' % r.returncode}",
          flush=True)

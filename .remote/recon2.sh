#!/bin/bash
echo "== WSL distro: $(uname -a)"
echo "== gpu in wsl =="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1
echo "== aigp repo =="
ls ~/aigp 2>/dev/null | head -16
echo "== venv torch =="
~/aigp/.venv/bin/python - <<'EOF' 2>&1 | tail -2
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
EOF
echo "== git state =="
cd ~/aigp 2>/dev/null && git log --oneline -3 && git status --short | head -5
echo "== data lake =="
ls ~/lrspeiser/ai-grand-prix 2>/dev/null | head -10
du -sh ~/lrspeiser/ai-grand-prix 2>/dev/null
echo "== home disk =="
df -h ~ | tail -1
echo "== running python =="
pgrep -a python | head -5

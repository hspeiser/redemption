#!/bin/bash
ssh -o ConnectTimeout=8 henry@gipsydanger '
echo "== GPU =="; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo "== procs =="; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
echo "== disk =="; df -h ~ | tail -1
echo "== aigp =="; ls ~/aigp 2>/dev/null | head -12
echo "== lake =="; ls ~/lrspeiser/ai-grand-prix 2>/dev/null | head -10
echo "== torch =="; ~/aigp/.venv/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))" 2>&1 | tail -1
echo "== git =="; cd ~/aigp 2>/dev/null && git log --oneline -3 2>/dev/null
'

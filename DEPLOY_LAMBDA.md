# Deploying on the Lambda A100 box (`lambda-a100` branch)

Target: `ssh ubuntu@141.148.191.11` — Ubuntu, **A100-SXM4-40GB**, 30 vCPUs, 200 GB RAM.
Persistent storage: `~/binglebob` → `/lambda/nfs/binglebob` (survives instance restarts).

**Layout**
- Code (ephemeral, re-deployable): `/home/ubuntu/redemption`
- Data + runs (persistent): `/home/ubuntu/binglebob/{datasets,runs}` — set in the configs.

**Torch:** A100 is Ampere (sm_80) → the default `cu124` wheels work (this branch does NOT
use the cu128 pin that the Blackwell 5090 needed).

---

## One-time setup
```bash
# 1. install uv
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

# 2. get the code (private repo, headless SSH can't auth GitHub -> ship a bundle)
#    From the local machine:  git bundle create lambda.bundle lambda-a100
#                             scp lambda.bundle ubuntu@141.148.191.11:
git clone -b lambda-a100 ~/lambda.bundle ~/redemption
cd ~/redemption
git remote set-url origin https://github.com/hspeiser/redemption.git

# 3. deps (pulls cu124 torch)
uv sync
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 4. persistent dirs
mkdir -p ~/binglebob/datasets ~/binglebob/runs
```

## Generate the dataset ONCE (persists on binglebob)
```bash
cd ~/redemption
nohup uv run python scripts/generate_data.py > ~/binglebob/gen.log 2>&1 &
# ~50k/5k/5k images on 28 workers; watch: tail -f ~/binglebob/gen.log
```

## Train — iterate model size with a ONE-LINE edit
Edit `configs/train.toml` → `[model].active`: `nano` → then `medium` → then `xpose`.
Each run trains on the same dataset (no regen) and writes to `~/binglebob/runs`.
```bash
cd ~/redemption
nohup uv run python scripts/train_model.py > ~/binglebob/train.log 2>&1 &
```
Detach note: `nohup ... &` survives SSH logout on Linux (unlike Windows).

## Live progress website
```bash
nohup uv run python scripts/serve_progress.py > ~/binglebob/serve.log 2>&1 &
```
View it via an SSH tunnel (reliable, no firewall changes):
```bash
ssh -L 8000:localhost:8000 ubuntu@141.148.191.11    # then open http://localhost:8000
```
Or open TCP 8000 in the Lambda firewall and browse `http://141.148.191.11:8000/`.

## Notes
- `[train].cache = "ram"` loads the dataset into RAM once so the NFS drive is read only
  at the start of each training run (fast thereafter).
- Checkpoints/best.pt persist under `~/binglebob/runs/...` across instance restarts.
- To re-deploy code after a change: re-bundle + scp, then `git pull ~/lambda.bundle lambda-a100`.

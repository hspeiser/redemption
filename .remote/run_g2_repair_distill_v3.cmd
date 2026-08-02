@echo off
setlocal
cd /d C:\Users\henry\aigp
set OUT=worldmodel\repairs\g2_ep2_distilled_actor_v3.pt
set LOG=worldmodel\repairs\g2_ep2_distilled_actor_v3.log
if exist "%OUT%" (
  echo Immutable output already exists: %OUT% > "%LOG%"
  exit /b 2
)
.venv-train\Scripts\python.exe scripts\distill_vq2_repair_actor.py ^
  --checkpoint worldmodel\repairs\g2_ep2_distill_inputs\protected_best.pt ^
  --repair-dataset worldmodel\repairs\g2_ep2_g1transition_dataset_v1\synthetic_actor_replay.npz ^
  --repair-manifest worldmodel\repairs\g2_ep2_g1transition_dataset_v1\manifest.json ^
  --repair-report worldmodel\repairs\g2_ep2_g1transition_search_v2\repair_search_report.json ^
  --anchor-replay worldmodel\repairs\g2_ep2_distill_inputs\protected_live_replay.npz ^
  --steps 8000 ^
  --batch-size 512 ^
  --learning-rate 0.00001 ^
  --repair-weight 0.25 ^
  --anchor-weight 1.0 ^
  --focus-anchor-gate 1 ^
  --focus-anchor-weight 4.0 ^
  --state-gate-hidden 32,16 ^
  --state-gate-steps 2000 ^
  --state-gate-false-positive-rate 0.001 ^
  --state-gate-min-repair-recall 0.95 ^
  --seed 20260828 ^
  --device cuda ^
  --out "%OUT%" > "%LOG%" 2>&1
exit /b %ERRORLEVEL%

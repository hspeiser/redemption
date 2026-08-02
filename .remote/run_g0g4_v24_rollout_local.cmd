@echo off
cd /d C:\Users\henry\Desktop\ai-gp
set PYTHONPATH=.
if not exist D:\ai-gp\worldmodel\v24_g0g4_rollout mkdir D:\ai-gp\worldmodel\v24_g0g4_rollout
.venv-train\Scripts\python.exe -u scripts\train_vq2_g0g4_worldmodel.py ^
  --dataset D:\ai-gp\worldmodel\g0g4_current_aug_19_impulse_clean ^
  --base-model data\fastsim_model_v2.json ^
  --map data\vq2_runtime_map_g9g15fix.json ^
  --init-ensemble D:\ai-gp\worldmodel\v23_g0g4_impulse_clean\residual_ensemble_v23.pt ^
  --out D:\ai-gp\worldmodel\v24_g0g4_rollout\residual_ensemble_v24.pt ^
  --members 5 ^
  --epochs 20 ^
  --batch-size 2048 ^
  --lr 0.0001 ^
  --rollout-finetune-epochs 3 ^
  --rollout-horizons 12,24,32 ^
  --rollout-lr 0.00002 ^
  --rollout-batch-size 512 ^
  --rollout-max-starts 8192 ^
  --seed 20260876 ^
  --device cuda ^
  1>D:\ai-gp\worldmodel\v24_g0g4_rollout\stdout.log ^
  2>D:\ai-gp\worldmodel\v24_g0g4_rollout\stderr.log
exit /b %ERRORLEVEL%

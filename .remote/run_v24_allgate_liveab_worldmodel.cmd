@echo off
cd /d C:\Users\henry\aigp
if not exist worldmodel\v24_allgate_liveab mkdir worldmodel\v24_allgate_liveab
.venv-train\Scripts\python.exe -u scripts\train_vq2_g0g4_worldmodel.py ^
  --dataset worldmodel\g0g16_master_currentera_v7_liveab ^
  --base-model data\fastsim_model_v2.json ^
  --map data\vq2_runtime_map_g9g15fix.json ^
  --init-ensemble worldmodel\v23_allgate_currentfinetune\residual_ensemble_v23.pt ^
  --out worldmodel\v24_allgate_liveab\residual_ensemble_v24.pt ^
  --members 5 ^
  --epochs 50 ^
  --batch-size 2048 ^
  --lr 2e-4 ^
  --device cuda ^
  1>worldmodel\v24_allgate_liveab\stdout.log ^
  2>worldmodel\v24_allgate_liveab\stderr.log

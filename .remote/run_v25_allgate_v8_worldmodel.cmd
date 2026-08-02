@echo off
cd /d C:\Users\henry\aigp
if not exist worldmodel\v25_allgate_v8 mkdir worldmodel\v25_allgate_v8
.venv-train\Scripts\python.exe -u scripts\train_vq2_g0g4_worldmodel.py ^
  --dataset worldmodel\g0g16_master_currentera_v8_v7 ^
  --base-model data\fastsim_model_v2.json ^
  --map data\vq2_runtime_map_g9g15fix.json ^
  --init-ensemble worldmodel\v24_allgate_liveab\residual_ensemble_v24.pt ^
  --out worldmodel\v25_allgate_v8\residual_ensemble_v25.pt ^
  --members 5 ^
  --epochs 50 ^
  --batch-size 2048 ^
  --lr 2e-4 ^
  --device cuda ^
  1>worldmodel\v25_allgate_v8\stdout.log ^
  2>worldmodel\v25_allgate_v8\stderr.log

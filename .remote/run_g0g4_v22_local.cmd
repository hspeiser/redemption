@echo off
cd /d C:\Users\henry\Desktop\ai-gp
if not exist D:\ai-gp\worldmodel\v22_g0g4_geometry_live mkdir D:\ai-gp\worldmodel\v22_g0g4_geometry_live
.venv-train\Scripts\python.exe -u scripts\train_vq2_g0g4_worldmodel.py ^
  --dataset D:\ai-gp\worldmodel\g0g4_current_aug_18_geometry_live ^
  --base-model data\fastsim_model_v2.json ^
  --map data\vq2_runtime_map_g9g15fix.json ^
  --init-ensemble worldmodel\v21_full16_baseline\residual_ensemble_v21.pt ^
  --out D:\ai-gp\worldmodel\v22_g0g4_geometry_live\residual_ensemble_v22.pt ^
  --members 5 ^
  --epochs 50 ^
  --batch-size 2048 ^
  --lr 0.0002 ^
  --seed 20260861 ^
  --device cuda ^
  1>D:\ai-gp\worldmodel\v22_g0g4_geometry_live\stdout.log ^
  2>D:\ai-gp\worldmodel\v22_g0g4_geometry_live\stderr.log
exit /b %ERRORLEVEL%

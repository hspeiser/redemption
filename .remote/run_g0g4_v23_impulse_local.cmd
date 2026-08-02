@echo off
cd /d C:\Users\henry\Desktop\ai-gp
if not exist D:\ai-gp\worldmodel\v23_g0g4_impulse_clean mkdir D:\ai-gp\worldmodel\v23_g0g4_impulse_clean
.venv-train\Scripts\python.exe -u scripts\train_vq2_g0g4_worldmodel.py ^
  --dataset D:\ai-gp\worldmodel\g0g4_current_aug_19_impulse_clean ^
  --base-model data\fastsim_model_v2.json ^
  --map data\vq2_runtime_map_g9g15fix.json ^
  --init-ensemble D:\ai-gp\worldmodel\v22_g0g4_geometry_live\residual_ensemble_v22.pt ^
  --out D:\ai-gp\worldmodel\v23_g0g4_impulse_clean\residual_ensemble_v23.pt ^
  --members 5 ^
  --epochs 50 ^
  --batch-size 2048 ^
  --lr 0.0002 ^
  --seed 20260865 ^
  --device cuda ^
  1>D:\ai-gp\worldmodel\v23_g0g4_impulse_clean\stdout.log ^
  2>D:\ai-gp\worldmodel\v23_g0g4_impulse_clean\stderr.log
exit /b %ERRORLEVEL%

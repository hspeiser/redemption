@echo off
cd /d C:\Users\henry\aigp
.venv-train\Scripts\python.exe -u scripts\compare_vq2_worldmodels.py ^
  --model v25=worldmodel\v25_allgate_v8\residual_ensemble_v25.pt ^
  --model v28=worldmodel\v28_allgate_registry_flywheel\residual_ensemble_v28.pt ^
  --model v30=worldmodel\v30_allgate_registry_flywheel\residual_ensemble_v30.pt ^
  --model v31=worldmodel\v31_allgate_registry_flywheel\residual_ensemble_v31.pt ^
  --dataset worldmodel\g0g16_allera_registry_audit_v2 ^
  --base-model data\fastsim_model_v2.json ^
  --map data\vq2_runtime_map_g9g15fix.json ^
  --out worldmodel\v31_allgate_registry_flywheel\fresh_registry_audit_h32.json ^
  --horizon 32 ^
  --device cuda ^
  1>worldmodel\v31_allgate_registry_flywheel\audit.stdout.log ^
  2>worldmodel\v31_allgate_registry_flywheel\audit.stderr.log
if errorlevel 1 (
  >worldmodel\v31_allgate_registry_flywheel\audit_exit_code.txt echo 1
  exit /b 1
)
>worldmodel\v31_allgate_registry_flywheel\audit_exit_code.txt echo 0
exit /b 0

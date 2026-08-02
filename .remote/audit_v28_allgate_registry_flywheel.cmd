@echo off
cd /d C:\Users\henry\aigp
.venv-train\Scripts\python.exe -u scripts\compare_vq2_worldmodels.py ^
  --model v25=worldmodel\v25_allgate_v8\residual_ensemble_v25.pt ^
  --model v28=worldmodel\v28_allgate_registry_flywheel\residual_ensemble_v28.pt ^
  --dataset worldmodel\g0g16_allera_registry_audit_v1 ^
  --base-model data\fastsim_model_v2.json ^
  --map data\vq2_runtime_map_g9g15fix.json ^
  --out worldmodel\v28_allgate_registry_flywheel\fresh_registry_audit_h32.json ^
  --horizon 32 ^
  --device cuda ^
  1>worldmodel\v28_allgate_registry_flywheel\audit.stdout.log ^
  2>worldmodel\v28_allgate_registry_flywheel\audit.stderr.log
echo %ERRORLEVEL%>worldmodel\v28_allgate_registry_flywheel\audit_exit_code.txt

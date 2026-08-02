@echo off
cd /d C:\Users\henry\aigp
if not exist worldmodel\v32_allgate_registry_flywheel mkdir worldmodel\v32_allgate_registry_flywheel
.venv-train\Scripts\python.exe -u scripts\train_vq2_g0g4_worldmodel.py ^
  --dataset worldmodel\g0g16_master_currentera_v32_registry ^
  --audit-dataset worldmodel\g0g16_allera_registry_audit_v2 ^
  --base-model data\fastsim_model_v2.json ^
  --map data\vq2_runtime_map_g9g15fix.json ^
  --init-ensemble worldmodel\v30_allgate_registry_flywheel\residual_ensemble_v30.pt ^
  --out worldmodel\v32_allgate_registry_flywheel\residual_ensemble_v32.pt ^
  --members 5 ^
  --epochs 25 ^
  --batch-size 2048 ^
  --lr 8e-5 ^
  --rollout-finetune-epochs 1 ^
  --rollout-horizons 8,16,32 ^
  --rollout-lr 2e-5 ^
  --rollout-batch-size 256 ^
  --rollout-max-starts 16384 ^
  --seed 20260832 ^
  --device cuda ^
  1>worldmodel\v32_allgate_registry_flywheel\stdout.log ^
  2>worldmodel\v32_allgate_registry_flywheel\stderr.log
if errorlevel 1 (
  >worldmodel\v32_allgate_registry_flywheel\exit_code.txt echo 1
  exit /b 1
)
>worldmodel\v32_allgate_registry_flywheel\exit_code.txt echo 0
exit /b 0

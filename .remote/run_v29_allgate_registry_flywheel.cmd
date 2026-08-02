@echo off
cd /d C:\Users\henry\aigp
if not exist worldmodel\v29_allgate_registry_flywheel mkdir worldmodel\v29_allgate_registry_flywheel
.venv-train\Scripts\python.exe -u scripts\train_vq2_g0g4_worldmodel.py ^
  --dataset worldmodel\g0g16_master_currentera_v29_registry ^
  --audit-dataset worldmodel\g0g16_allera_registry_audit_v1 ^
  --base-model data\fastsim_model_v2.json ^
  --map data\vq2_runtime_map_g9g15fix.json ^
  --init-ensemble worldmodel\v28_allgate_registry_flywheel\residual_ensemble_v28.pt ^
  --out worldmodel\v29_allgate_registry_flywheel\residual_ensemble_v29.pt ^
  --members 5 ^
  --epochs 25 ^
  --batch-size 2048 ^
  --lr 8e-5 ^
  --rollout-finetune-epochs 1 ^
  --rollout-horizons 8,16,32 ^
  --rollout-lr 2e-5 ^
  --rollout-batch-size 256 ^
  --rollout-max-starts 16384 ^
  --seed 20260829 ^
  --device cuda ^
  1>worldmodel\v29_allgate_registry_flywheel\stdout.log ^
  2>worldmodel\v29_allgate_registry_flywheel\stderr.log
echo %ERRORLEVEL%>worldmodel\v29_allgate_registry_flywheel\exit_code.txt

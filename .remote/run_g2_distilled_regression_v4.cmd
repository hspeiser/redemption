@echo off
setlocal EnableDelayedExpansion
cd /d C:\Users\henry\aigp
set OUT=worldmodel\repairs\g2_ep2_distilled_regression_v4
set LOG=worldmodel\repairs\g2_ep2_distilled_regression_v4.log
if exist "%OUT%" (
  echo Immutable output already exists: %OUT% > "%LOG%"
  exit /b 2
)
mkdir "%OUT%"
for %%M in (v23 v24 v25) do (
  if "%%M"=="v23" (set ENS=worldmodel\v23_allgate_currentfinetune\residual_ensemble_v23.pt& set SEED=20261123)
  if "%%M"=="v24" (set ENS=worldmodel\v24_allgate_liveab\residual_ensemble_v24.pt& set SEED=20261124)
  if "%%M"=="v25" (set ENS=worldmodel\v25_allgate_v8\residual_ensemble_v25.pt& set SEED=20261125)
  .venv-train\Scripts\python.exe scripts\audit_vq2_residual_candidate.py ^
    --actor worldmodel\repairs\g2_ep2_distilled_actor_v2.pt ^
    --ensemble !ENS! ^
    --model data\fastsim_model_v2.json ^
    --controller-model data\fastsim_model_v2.json ^
    --demo data\vq2_g0g1fast9267_g2plus_clean_demo.npz ^
    --map data\vq2_runtime_map_g9g15fix.json ^
    --obstacles data\vq2_obstacles_inflated.json ^
    --teacher-config worldmodel\repairs\g2_ep2_distill_inputs\candidate_config_phase075.json ^
    --out "%OUT%\%%M_candidate.json" ^
    --worlds 4096 ^
    --seed !SEED! ^
    --aleatoric-scale 1.5 ^
    --impulse-rate-hz 0.08 ^
    --residual-scale 0.25 ^
    --multigate-estimator ^
    --device cuda >> "%LOG%" 2>&1
  if errorlevel 1 exit /b !errorlevel!
)
exit /b 0

@echo off
setlocal EnableDelayedExpansion
cd /d C:\Users\henry\aigp
set OUT=worldmodel\repairs\g2_ep2_distilled_regression_v2
set LOG=worldmodel\repairs\g2_ep2_distilled_regression_v2.log
if exist "%OUT%" (
  echo Immutable output already exists: %OUT% > "%LOG%"
  exit /b 2
)
mkdir "%OUT%"
for %%M in (v23 v24 v25) do (
  if "%%M"=="v23" (set ENS=worldmodel\v23_allgate_currentfinetune\residual_ensemble_v23.pt& set SEED=20261023)
  if "%%M"=="v24" (set ENS=worldmodel\v24_allgate_liveab\residual_ensemble_v24.pt& set SEED=20261024)
  if "%%M"=="v25" (set ENS=worldmodel\v25_allgate_v8\residual_ensemble_v25.pt& set SEED=20261025)
  call :audit candidate %%M worldmodel\repairs\g2_ep2_distill_inputs\candidate_config.json
  if errorlevel 1 exit /b !errorlevel!
  call :audit geometry_only %%M worldmodel\repairs\g2_ep2_distill_inputs\candidate_config.json --protected-base
  if errorlevel 1 exit /b !errorlevel!
  call :audit protected_source %%M worldmodel\repairs\g2_ep2_source_config.json --protected-base
  if errorlevel 1 exit /b !errorlevel!
)
exit /b 0

:audit
set ARM=%1
set MODEL_LABEL=%2
set CFG=%3
set EXTRA=%4
.venv-train\Scripts\python.exe scripts\audit_vq2_residual_candidate.py ^
  --actor worldmodel\repairs\g2_ep2_distilled_actor_v1.pt ^
  --ensemble !ENS! ^
  --model data\fastsim_model_v2.json ^
  --controller-model data\fastsim_model_v2.json ^
  --demo data\vq2_g0g1fast9267_g2plus_clean_demo.npz ^
  --map data\vq2_runtime_map_g9g15fix.json ^
  --obstacles data\vq2_obstacles_inflated.json ^
  --teacher-config %CFG% ^
  --out "%OUT%\%MODEL_LABEL%_%ARM%.json" ^
  --worlds 4096 ^
  --seed !SEED! ^
  --aleatoric-scale 1.5 ^
  --impulse-rate-hz 0.08 ^
  --residual-scale 0.25 ^
  --multigate-estimator ^
  --device cuda %EXTRA% >> "%LOG%" 2>&1
exit /b %ERRORLEVEL%

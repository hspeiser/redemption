@echo off
setlocal
cd /d C:\Users\henry\aigp
set COMMON=--base-model data\fastsim_model_v2.json --demo data\vq2_g0g1fast9267_g2plus_clean_demo.npz --map data\vq2_runtime_map_g9g15fix.json --obstacles data\vq2_obstacles_inflated.json --worlds 2048 --aleatoric-scale 1.5 --position-sigma-floor-m 0.10 --minimum-family-rate 0.20 --device cuda

.venv-train\Scripts\python.exe -u scripts\reproduce_vq2_failure.py --snapshot-artifact worldmodel\repairs\flywheel_g5_ep1_20260802_v1 --ensemble worldmodel\v25_allgate_v8\residual_ensemble_v25.pt %COMMON% --seed 20260851 --out worldmodel\repairs\flywheel_g5_ep1_20260802_v1\reproduction_v25_racefix_v2
if errorlevel 1 exit /b %ERRORLEVEL%
.venv-train\Scripts\python.exe -u scripts\reproduce_vq2_failure.py --snapshot-artifact worldmodel\repairs\flywheel_g5_ep1_20260802_v1 --ensemble worldmodel\v28_allgate_registry_flywheel\residual_ensemble_v28.pt %COMMON% --seed 20260851 --out worldmodel\repairs\flywheel_g5_ep1_20260802_v1\reproduction_v28_racefix_v2
if errorlevel 1 exit /b %ERRORLEVEL%
.venv-train\Scripts\python.exe -u scripts\reproduce_vq2_failure.py --snapshot-artifact worldmodel\repairs\flywheel_g6_ep0_20260802_v1 --ensemble worldmodel\v25_allgate_v8\residual_ensemble_v25.pt %COMMON% --seed 20260861 --out worldmodel\repairs\flywheel_g6_ep0_20260802_v1\reproduction_v25_racefix_v2
if errorlevel 1 exit /b %ERRORLEVEL%
.venv-train\Scripts\python.exe -u scripts\reproduce_vq2_failure.py --snapshot-artifact worldmodel\repairs\flywheel_g6_ep0_20260802_v1 --ensemble worldmodel\v28_allgate_registry_flywheel\residual_ensemble_v28.pt %COMMON% --seed 20260861 --out worldmodel\repairs\flywheel_g6_ep0_20260802_v1\reproduction_v28_racefix_v2

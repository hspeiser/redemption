@echo off
setlocal
cd /d C:\Users\henry\aigp
set OUT=worldmodel\repairs\g2_ep2_geometry_search_v1
set LOG=worldmodel\repairs\g2_ep2_geometry_search_v1.log
if exist "%OUT%" (
  echo Immutable output already exists: %OUT% > "%LOG%"
  exit /b 2
)
.venv-train\Scripts\python.exe scripts\search_vq2_counterfactual_repair.py ^
  --snapshot-artifact worldmodel\repairs\g2_ep2_20260801_snapshot_v1 ^
  --reproduction-report worldmodel\repairs\g2_ep2_reproduction_v2\reproduction_report.json ^
  --ensemble worldmodel\v23_allgate_currentfinetune\residual_ensemble_v23.pt ^
  --ensemble worldmodel\v24_allgate_liveab\residual_ensemble_v24.pt ^
  --ensemble worldmodel\v25_allgate_v8\residual_ensemble_v25.pt ^
  --base-model data\fastsim_model_v2.json ^
  --demo data\vq2_g0g1fast9267_g2plus_clean_demo.npz ^
  --map data\vq2_runtime_map_g9g15fix.json ^
  --obstacles data\vq2_obstacles_inflated.json ^
  --teacher-config worldmodel\repairs\g2_ep2_source_config.json ^
  --population 64 ^
  --elite 10 ^
  --iterations 14 ^
  --worlds 32 ^
  --selection-worlds 128 ^
  --audit-worlds 512 ^
  --knots 5 ^
  --authority 0.4 ^
  --geometry-search ^
  --max-lateral-offset 0.35 ^
  --max-vertical-offset 0.25 ^
  --max-speed-scale-delta 0.15 ^
  --post-steps 24 ^
  --aleatoric-scale 1.0 ^
  --max-support-p90 3.0 ^
  --max-disagreement-p90 0.08 ^
  --seed 20260820 ^
  --device cuda ^
  --out "%OUT%" > "%LOG%" 2>&1
exit /b %ERRORLEVEL%

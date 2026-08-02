@echo off
cd /d C:\Users\henry\aigp
if not exist worldmodel\v25_residual_smoke_multigate mkdir worldmodel\v25_residual_smoke_multigate
.venv-train\Scripts\python.exe -u scripts\optimize_vq2_residual_schedule.py ^
  --actor worldmodel\live_champion_v79_best.pt ^
  --ensemble worldmodel\v23_allgate_currentfinetune\residual_ensemble_v23.pt ^
  --ensemble worldmodel\v24_allgate_liveab\residual_ensemble_v24.pt ^
  --ensemble worldmodel\v25_allgate_v8\residual_ensemble_v25.pt ^
  --model data\fastsim_model_v2.json ^
  --controller-model data\fastsim_model_v2.json ^
  --demo data\vq2_g0g1fast9267_g2plus_clean_demo.npz ^
  --map data\vq2_runtime_map_g9g15fix.json ^
  --obstacles data\vq2_obstacles_inflated.json ^
  --teacher-config worldmodel\teacher_full17_cem_safe_v1.json ^
  --residual-scale 0.25 ^
  --multigate-estimator ^
  --aleatoric-scale 1.5 ^
  --impulse-rate-hz 0.08 ^
  --reliability-floor 0.97 ^
  --seed 20260802 ^
  --smoke ^
  --out worldmodel\v25_residual_smoke_multigate\result.json ^
  1>worldmodel\v25_residual_smoke_multigate\stdout.log ^
  2>worldmodel\v25_residual_smoke_multigate\stderr.log

@echo off
cd /d C:\Users\henry\aigp
.venv-train\Scripts\python.exe -u scripts\build_vq2_offline_live_calibration.py ^
  --training-root worldmodel\calibration_bundle_v1 ^
  --fallback-actor worldmodel\live_champion_v79_best.pt ^
  --ensemble worldmodel\v25_allgate_v8\residual_ensemble_v25.pt ^
  --model data\fastsim_model_v2.json ^
  --controller-model data\fastsim_model_v2.json ^
  --demo data\vq2_g0g1fast9267_g2plus_clean_demo.npz ^
  --map data\vq2_runtime_map_g9g15fix.json ^
  --obstacles data\vq2_obstacles_inflated.json ^
  --worlds 256 ^
  --seed 20260803 ^
  --aleatoric-scale 1.5 ^
  --impulse-rate-hz 0.08 ^
  --out worldmodel\calibration_v25_v1.json ^
  1>worldmodel\calibration_v25_v1.stdout.log ^
  2>worldmodel\calibration_v25_v1.stderr.log

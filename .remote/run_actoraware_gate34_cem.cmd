@echo off
cd /d C:\Users\henry\aigp
.venv-train\Scripts\python.exe -u scripts\optimize_vq2_g0g4_worldmodel.py ^
  --dataset worldmodel\g0g4_current_aug_17_full16_baseline ^
  --ensemble worldmodel\multimodel_seqg2_probe\residual_ensemble_current_v13.pt ^
  --ensemble worldmodel\multimodel_seqg2_probe\residual_ensemble_v19.pt ^
  --ensemble worldmodel\v21_full16_baseline\residual_ensemble_v20.pt ^
  --ensemble worldmodel\v21_full16_baseline\residual_ensemble_v21.pt ^
  --model data\fastsim_model_v2.json ^
  --controller-model data\fastsim_model_v2.json ^
  --demo data\vq2_g0g1fast9267_g2plus_clean_demo.npz ^
  --map data\vq2_runtime_map_g9g15fix.json ^
  --obstacles data\vq2_obstacles_inflated.json ^
  --teacher-config worldmodel\teacher_full17_cem_safe_v1.json ^
  --initial worldmodel\exactparity_multimodel_tier87_cem_v1.json ^
  --actor worldmodel\ppo_multimodel_segmentcredit_v8\best.pt ^
  --residual-scale 0.20 ^
  --active-gates 3 4 ^
  --multigate-vision ^
  --population 32 ^
  --elite 8 ^
  --iterations 12 ^
  --worlds 64 ^
  --selection-worlds 1024 ^
  --aleatoric-scale 1.25 ^
  --impulse-rate-hz 0.06 ^
  --live-estimator-realism ^
  --reliability-floor 0.90 ^
  --time-weight 50 ^
  --tier-target 8.7 ^
  --tier-bonus 250 ^
  --device cuda ^
  --out worldmodel\actoraware_gate34_cem_v1.json ^
  %*

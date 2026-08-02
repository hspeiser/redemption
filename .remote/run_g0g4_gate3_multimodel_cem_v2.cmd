@echo off
cd /d C:\Users\henry\Desktop\ai-gp
set PYTHONPATH=.
.venv-train\Scripts\python.exe -u scripts\optimize_vq2_g0g4_worldmodel.py ^
  --dataset D:\ai-gp\worldmodel\g0g4_current_aug_19_impulse_clean ^
  --ensemble D:\ai-gp\worldmodel\v22_g0g4_geometry_live\residual_ensemble_v22.pt ^
  --ensemble D:\ai-gp\worldmodel\v23_g0g4_impulse_clean\residual_ensemble_v23.pt ^
  --ensemble D:\ai-gp\worldmodel\v24_g0g4_rollout\residual_ensemble_v24.pt ^
  --model data\fastsim_model_v2.json ^
  --controller-model data\fastsim_model_v2.json ^
  --demo data\vq2_g0g1fast9267_g2plus_clean_demo.npz ^
  --map data\vq2_runtime_map_g9g15fix.json ^
  --obstacles data\vq2_obstacles_inflated.json ^
  --teacher-config D:\ai-gp\worldmodel\teacher_full17_cem_safe_v1.json ^
  --initial D:\ai-gp\worldmodel\g0g4_v22only_tier87_cem_v1.json ^
  --actor worldmodel\ppo_multimodel_segmentcredit_v8\best.pt ^
  --residual-scale 0.20 ^
  --active-gates 3 ^
  --geometry-gates 3 ^
  --geometry-limit 0.40 ^
  --multigate-vision ^
  --population 48 ^
  --elite 10 ^
  --iterations 24 ^
  --worlds 48 ^
  --selection-worlds 2048 ^
  --final-worlds 4096 ^
  --clean-worlds 1024 ^
  --aleatoric-scale 1.5 ^
  --impulse-rate-hz 0.08 ^
  --live-estimator-realism ^
  --reliability-floor 0.90 ^
  --time-weight 60 ^
  --tier-target 8.7 ^
  --tier-bonus 300 ^
  --seed 20260931 ^
  --device cuda ^
  --out D:\ai-gp\worldmodel\g0g4_gate3_multimodel_cem_v2.json ^
  1>D:\ai-gp\worldmodel\g0g4_gate3_multimodel_cem_v2.stdout.log ^
  2>D:\ai-gp\worldmodel\g0g4_gate3_multimodel_cem_v2.stderr.log
exit /b %ERRORLEVEL%

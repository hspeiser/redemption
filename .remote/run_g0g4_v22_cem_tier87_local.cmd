@echo off
cd /d C:\Users\henry\Desktop\ai-gp
.venv-train\Scripts\python.exe -u scripts\optimize_vq2_g0g4_worldmodel.py ^
  --dataset D:\ai-gp\worldmodel\g0g4_current_aug_18_geometry_live ^
  --ensemble D:\ai-gp\worldmodel\v22_g0g4_geometry_live\residual_ensemble_v22.pt ^
  --ensemble worldmodel\multimodel_seqg2_probe\residual_ensemble_current_v13.pt ^
  --ensemble worldmodel\multimodel_seqg2_probe\residual_ensemble_v19.pt ^
  --ensemble worldmodel\v21_full16_baseline\residual_ensemble_v21.pt ^
  --model data\fastsim_model_v2.json ^
  --controller-model data\fastsim_model_v2.json ^
  --demo data\vq2_g0g1fast9267_g2plus_clean_demo.npz ^
  --map data\vq2_runtime_map_g9g15fix.json ^
  --obstacles data\vq2_obstacles_inflated.json ^
  --teacher-config D:\ai-gp\worldmodel\teacher_full17_cem_safe_v1.json ^
  --initial D:\ai-gp\worldmodel\g0g4_livecal_actoraware_tier87_cem_local_v5.json ^
  --actor worldmodel\ppo_multimodel_segmentcredit_v8\best.pt ^
  --residual-scale 0.20 ^
  --active-gates 1 2 3 4 ^
  --geometry-gates 1 2 3 4 ^
  --geometry-limit 0.35 ^
  --multigate-vision ^
  --population 64 ^
  --elite 12 ^
  --iterations 24 ^
  --worlds 64 ^
  --selection-worlds 1024 ^
  --final-worlds 2048 ^
  --clean-worlds 512 ^
  --aleatoric-scale 1.5 ^
  --impulse-rate-hz 0.08 ^
  --live-estimator-realism ^
  --reliability-floor 0.90 ^
  --time-weight 60 ^
  --tier-target 8.7 ^
  --tier-bonus 300 ^
  --seed 20260863 ^
  --device cuda ^
  --out D:\ai-gp\worldmodel\g0g4_v22_multimodel_tier87_cem_v1.json ^
  1>D:\ai-gp\worldmodel\g0g4_v22_multimodel_tier87_cem_v1.stdout.log ^
  2>D:\ai-gp\worldmodel\g0g4_v22_multimodel_tier87_cem_v1.stderr.log
exit /b %ERRORLEVEL%

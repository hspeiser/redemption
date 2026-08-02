@echo off
setlocal
cd /d C:\Users\henry\aigp
.venv-train\Scripts\python.exe -u scripts\search_vq2_counterfactual_repair.py ^
  --snapshot-artifact worldmodel\repairs\flywheel_g6_ep0_20260802_v1 ^
  --reproduction-report worldmodel\repairs\flywheel_g6_ep0_20260802_v1\reproduction_v25_racefix_v2\reproduction_report.json ^
  --only-rollback-steps 12,21,30,45 ^
  --ensemble worldmodel\v25_allgate_v8\residual_ensemble_v25.pt ^
  --ensemble worldmodel\v28_allgate_registry_flywheel\residual_ensemble_v28.pt ^
  --base-model data\fastsim_model_v2.json ^
  --demo data\vq2_g0g1fast9267_g2plus_clean_demo.npz ^
  --map data\vq2_runtime_map_g9g15fix.json ^
  --obstacles data\vq2_obstacles_inflated.json ^
  --teacher-config worldmodel\teacher_full17_cem_safe_v1.json ^
  --population 64 ^
  --elite 8 ^
  --iterations 12 ^
  --worlds 32 ^
  --selection-worlds 256 ^
  --audit-worlds 1024 ^
  --knots 6 ^
  --authority 0.12 ^
  --post-steps 30 ^
  --aleatoric-scale 1.5 ^
  --max-support-p90 3.0 ^
  --max-disagreement-p90 0.10 ^
  --seed 20260862 ^
  --device cuda ^
  --out worldmodel\repairs\flywheel_g6_action_repair_v1

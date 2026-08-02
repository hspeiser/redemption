@echo off
cd /d C:\Users\henry\Desktop\ai-gp
set PYTHONPATH=.
set COMMON=--dataset D:\ai-gp\worldmodel\g0g4_current_aug_19_impulse_clean --ensemble D:\ai-gp\worldmodel\v22_g0g4_geometry_live\residual_ensemble_v22.pt --ensemble D:\ai-gp\worldmodel\v23_g0g4_impulse_clean\residual_ensemble_v23.pt --ensemble D:\ai-gp\worldmodel\v24_g0g4_rollout\residual_ensemble_v24.pt --model data\fastsim_model_v2.json --controller-model data\fastsim_model_v2.json --actor worldmodel\ppo_multimodel_segmentcredit_v8\best.pt --demo data\vq2_g0g1fast9267_g2plus_clean_demo.npz --map data\vq2_runtime_map_g9g15fix.json --obstacles data\vq2_obstacles_inflated.json --teacher-config D:\ai-gp\worldmodel\teacher_full17_cem_safe_v1.json --worlds 2048 --seed 20260917 --device cuda --aleatoric-scale 1.5 --impulse-rate-hz 0.08 --residual-scale 0.20 --geometry-limit 0.40 --tier-target 8.7 --multigate-vision --live-estimator-realism

.venv-train\Scripts\python.exe -u scripts\audit_vq2_g0g4_candidate.py --candidate D:\ai-gp\worldmodel\g0g4_v22only_tier87_cem_v1.json %COMMON% --out D:\ai-gp\worldmodel\g0g4_gate3_offset_screen_v1_baseline.json 1>D:\ai-gp\worldmodel\g0g4_gate3_offset_screen_v1_baseline.stdout.log 2>D:\ai-gp\worldmodel\g0g4_gate3_offset_screen_v1_baseline.stderr.log
if errorlevel 1 exit /b %ERRORLEVEL%

.venv-train\Scripts\python.exe -u scripts\audit_vq2_g0g4_candidate.py --candidate D:\ai-gp\worldmodel\g0g4_gate3_left035_candidate_v1.json %COMMON% --out D:\ai-gp\worldmodel\g0g4_gate3_offset_screen_v1_left035.json 1>D:\ai-gp\worldmodel\g0g4_gate3_offset_screen_v1_left035.stdout.log 2>D:\ai-gp\worldmodel\g0g4_gate3_offset_screen_v1_left035.stderr.log
exit /b %ERRORLEVEL%

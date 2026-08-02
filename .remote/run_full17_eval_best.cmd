@echo off
cd /d C:\Users\henry\aigp
.venv-train\Scripts\python.exe -u scripts\fastsim_train_ppo.py ^
  --eval-only ^
  --n-envs 2048 ^
  --eval-envs 2048 ^
  --eval-seed 20260803 ^
  --run-dir worldmodel\ppo_multimodel_segmentcredit_v8\full17_v23_audit ^
  --resume worldmodel\ppo_multimodel_segmentcredit_v8\best.pt ^
  --rate-sign 1 ^
  --model data\fastsim_model_v2.json ^
  --map data\vq2_runtime_map_g9g15fix.json ^
  --demo-npz data\vq2_g0g1fast9267_g2plus_clean_demo.npz ^
  --bc-init data\vq2_g0g1fast9267_g2plus_clean_demo.npz ^
  --residual ^
  --residual-scale 0.20 ^
  --active-residual-gates 0 1 2 3 4 ^
  --live-teacher-config worldmodel\teacher_full17_cem_safe_v1.json ^
  --controller-model data\fastsim_model_v2.json ^
  --race-gates 17 ^
  --max-episode-s 45 ^
  --random-start-frac 0 ^
  --world-model ^
    worldmodel\v22_allgate_broad\residual_ensemble_v22.pt ^
    worldmodel\v23_allgate_currentfinetune\residual_ensemble_v23.pt ^
  --world-model-aleatoric-scale 1.25 ^
  --noise-era 10hz ^
  --multigate-vision ^
  --spawn-at-rest ^
  --action-smoothness 0.12 ^
  --act-delay-min 0 ^
  --act-delay-max 0 ^
  --dr-thrust 0.99 1.01 ^
  --dr-rate-gain 0.99 1.01 ^
  --dr-rate-tau 0.97 1.03 ^
  --dr-drag 0.23 0.27 ^
  --eval-gate-masks configured;none

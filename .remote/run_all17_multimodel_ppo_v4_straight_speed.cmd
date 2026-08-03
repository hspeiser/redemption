@echo off
setlocal
cd /d C:\Users\henry\aigp

set RUN=worldmodel\ppo_all17_multimodel_v4_straight_speed
if not exist "%RUN%" mkdir "%RUN%"

.venv-train\Scripts\python.exe -u scripts\fastsim_train_ppo.py ^
  --iters 431 ^
  --n-envs 16 ^
  --horizon 32 ^
  --lr 2e-5 ^
  --epochs 4 ^
  --minibatch 512 ^
  --gamma 0.999 ^
  --lam 0.995 ^
  --entropy 5e-5 ^
  --device cuda ^
  --run-dir "%RUN%" ^
  --rate-sign 1 ^
  --model data\fastsim_model_v2.json ^
  --map data\vq2_runtime_map_g9g15fix.json ^
  --demo-npz data\vq2_g0g1fast9267_g2plus_clean_demo.npz ^
  --bc-init data\vq2_g0g1fast9267_g2plus_clean_demo.npz ^
  --bc-steps 0 ^
  --bc-anchor 0.03 ^
  --residual ^
  --residual-scale 0.20 ^
  --active-residual-gates 11 12 13 14 15 16 ^
  --residual-log-std -2.8 ^
  --resume worldmodel\ppo_all17_multimodel_v3_late_safe\best.pt ^
  --resume-reset-optimizer ^
  --resume-log-std -2.8 ^
  --live-teacher-config worldmodel\teacher_full17_cem_safe_v1.json ^
  --controller-model data\fastsim_model_v2.json ^
  --race-gates 17 ^
  --max-episode-s 45 ^
  --random-start-frac 0.90 ^
  --start-gate-weights 1 1 1 1 1 1 1 1 1 1 3 14 14 12 7 7 9 ^
  --world-model ^
    worldmodel\v25_allgate_v8\residual_ensemble_v25.pt ^
    worldmodel\v28_allgate_registry_flywheel\residual_ensemble_v28.pt ^
    worldmodel\v30_allgate_registry_flywheel\residual_ensemble_v30.pt ^
  --world-model-aleatoric-scale 1.25 ^
  --noise-era 10hz ^
  --multigate-vision ^
  --spawn-at-rest ^
  --impulse-rate-hz 0.04 ^
  --impulse-min-mps 0.10 ^
  --impulse-max-mps 0.55 ^
  --impulse-vertical-scale 0.35 ^
  --action-smoothness 0.12 ^
  --act-delay-min 0 ^
  --act-delay-max 0 ^
  --dr-thrust 0.99 1.01 ^
  --dr-rate-gain 0.99 1.01 ^
  --dr-rate-tau 0.97 1.03 ^
  --dr-drag 0.23 0.27 ^
  --time-penalty-per-s 3.0 ^
  --collision-penalty 120.0 ^
  --clearance-bonus 35.0 ^
  --finish-time-target-s 34.5 ^
  --finish-time-bonus-per-s 60.0 ^
  --gate-time-targets 2.30 1.97 1.10 1.57 2.40 2.43 1.80 2.07 1.53 2.07 2.10 2.80 2.70 2.65 1.80 2.20 2.30 ^
  --gate-time-bonus-per-s 35.0 ^
  --demo-crossing-bonus 12.0 ^
  --demo-crossing-radius-m 0.55 ^
  --eval-interval 10 ^
  --eval-envs 16 ^
  --eval-seed 20260829 ^
  1>"%RUN%\stdout.log" ^
  2>"%RUN%\stderr.log"

exit /b %ERRORLEVEL%

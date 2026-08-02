@echo off
cd /d C:\Users\henry\aigp
if not exist worldmodel\ppo_all17_multimodel_v3_late_safe mkdir worldmodel\ppo_all17_multimodel_v3_late_safe
.venv-train\Scripts\python.exe -u scripts\fastsim_train_ppo.py ^
  --iters 500 ^
  --n-envs 1024 ^
  --horizon 256 ^
  --lr 4e-5 ^
  --epochs 4 ^
  --minibatch 8192 ^
  --gamma 0.999 ^
  --lam 0.995 ^
  --entropy 1e-4 ^
  --device cuda ^
  --run-dir worldmodel\ppo_all17_multimodel_v3_late_safe ^
  --rate-sign 1 ^
  --model data\fastsim_model_v2.json ^
  --map data\vq2_runtime_map_g9g15fix.json ^
  --demo-npz data\vq2_g0g1fast9267_g2plus_clean_demo.npz ^
  --bc-init data\vq2_g0g1fast9267_g2plus_clean_demo.npz ^
  --bc-steps 0 ^
  --bc-anchor 0.02 ^
  --residual ^
  --residual-scale 0.20 ^
  --active-residual-gates 6 7 8 9 10 11 12 13 14 15 16 ^
  --residual-log-std -2.6 ^
  --resume worldmodel\ppo_all17_multimodel_v2\best.pt ^
  --resume-reset-optimizer ^
  --resume-log-std -2.6 ^
  --live-teacher-config worldmodel\teacher_full17_cem_safe_v1.json ^
  --controller-model data\fastsim_model_v2.json ^
  --race-gates 17 ^
  --max-episode-s 50 ^
  --random-start-frac 0.85 ^
  --start-gate-weights 1 1 1 1 1 1 6 6 6 6 4 6 4 5 4 4 5 ^
  --world-model ^
    worldmodel\v22_allgate_broad\residual_ensemble_v22.pt ^
    worldmodel\v23_allgate_currentfinetune\residual_ensemble_v23.pt ^
    worldmodel\v24_allgate_liveab\residual_ensemble_v24.pt ^
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
  --time-penalty-per-s 2.0 ^
  --collision-penalty 100.0 ^
  --clearance-bonus 30.0 ^
  --finish-time-target-s 38.5 ^
  --finish-time-bonus-per-s 40.0 ^
  --gate-time-targets 2.30 1.97 1.10 1.57 2.40 2.43 1.80 2.07 1.53 2.07 2.10 3.70 3.47 3.10 1.80 2.33 2.50 ^
  --gate-time-bonus-per-s 20.0 ^
  --demo-crossing-bonus 10.0 ^
  --demo-crossing-radius-m 0.55 ^
  --eval-interval 10 ^
  --eval-envs 1024 ^
  --eval-seed 20260806 ^
  1>worldmodel\ppo_all17_multimodel_v3_late_safe\stdout.log ^
  2>worldmodel\ppo_all17_multimodel_v3_late_safe\stderr.log

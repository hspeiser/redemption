@echo off
cd /d C:\Users\henry\aigp
.venv-train\Scripts\python.exe -u scripts\fastsim_train_ppo.py ^
  --iters 400 ^
  --n-envs 2048 ^
  --horizon 128 ^
  --lr 1e-4 ^
  --epochs 4 ^
  --minibatch 16384 ^
  --gamma 0.999 ^
  --lam 0.995 ^
  --entropy 1e-4 ^
  --device cuda ^
  --run-dir worldmodel\ppo_multimodel_segmentcredit_v8 ^
  --rate-sign 1 ^
  --model data\fastsim_model_v2.json ^
  --map data\vq2_runtime_map_g9g15fix.json ^
  --demo-npz data\vq2_g0g1fast9267_g2plus_clean_demo.npz ^
  --bc-init data\vq2_g0g1fast9267_g2plus_clean_demo.npz ^
  --bc-steps 0 ^
  --bc-anchor 0.005 ^
  --residual ^
  --residual-scale 0.20 ^
  --residual-log-std -2.4 ^
  --resume worldmodel\ppo_multimodel_longcredit_v7\best.pt ^
  --resume-reset-optimizer ^
  --resume-log-std -2.4 ^
  --live-teacher-config worldmodel\teacher_full17_cem_safe_v1.json ^
  --controller-model data\fastsim_model_v2.json ^
  --race-gates 5 ^
  --max-episode-s 12 ^
  --random-start-frac 0 ^
  --start-gate-weights 4 3 3 3 3 ^
  --world-model ^
    worldmodel\multimodel_seqg2_probe\residual_ensemble_current_v13.pt ^
    worldmodel\multimodel_seqg2_probe\residual_ensemble_v19.pt ^
    worldmodel\v21_full16_baseline\residual_ensemble_v20.pt ^
    worldmodel\v21_full16_baseline\residual_ensemble_v21.pt ^
  --world-model-aleatoric-scale 1.25 ^
  --noise-era 10hz ^
  --multigate-vision ^
  --spawn-at-rest ^
  --impulse-rate-hz 0.06 ^
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
  --finish-time-target-s 9.5 ^
  --finish-time-bonus-per-s 200.0 ^
  --gate-time-targets 2.3 1.966667 1.133333 1.6 2.4 ^
  --gate-time-bonus-per-s 200.0 ^
  --demo-crossing-bonus 2.0 ^
  --demo-crossing-radius-m 0.50 ^
  --eval-interval 5 ^
  --eval-envs 2048 ^
  --eval-seed 20260802 ^
  1>worldmodel\ppo_multimodel_segmentcredit_v8\stdout.log ^
  2>worldmodel\ppo_multimodel_segmentcredit_v8\stderr.log

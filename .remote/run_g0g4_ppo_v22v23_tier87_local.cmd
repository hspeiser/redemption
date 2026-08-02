@echo off
cd /d C:\Users\henry\Desktop\ai-gp
set PYTHONPATH=.
.venv-train\Scripts\python.exe -u scripts\fastsim_train_ppo.py ^
  --iters 400 ^
  --n-envs 2048 ^
  --horizon 128 ^
  --lr 5e-5 ^
  --epochs 4 ^
  --minibatch 16384 ^
  --gamma 0.999 ^
  --lam 0.995 ^
  --entropy 1e-4 ^
  --device cuda ^
  --run-dir D:\ai-gp\worldmodel\ppo_g0g4_v22v23_tier87_v1 ^
  --rate-sign 1 ^
  --model data\fastsim_model_v2.json ^
  --map data\vq2_runtime_map_g9g15fix.json ^
  --demo-npz data\vq2_g0g1fast9267_g2plus_clean_demo.npz ^
  --bc-init data\vq2_g0g1fast9267_g2plus_clean_demo.npz ^
  --bc-steps 0 ^
  --bc-anchor 0.005 ^
  --residual ^
  --residual-scale 0.20 ^
  --active-residual-gates 0 1 2 3 4 ^
  --residual-log-std -2.6 ^
  --resume worldmodel\ppo_multimodel_segmentcredit_v8\best.pt ^
  --resume-reset-optimizer ^
  --resume-log-std -2.6 ^
  --live-teacher-config D:\ai-gp\worldmodel\teacher_g0g4_v22cem_v1.json ^
  --controller-model data\fastsim_model_v2.json ^
  --race-gates 5 ^
  --max-episode-s 12 ^
  --random-start-frac 0 ^
  --world-model ^
    D:\ai-gp\worldmodel\v22_g0g4_geometry_live\residual_ensemble_v22.pt ^
    D:\ai-gp\worldmodel\v23_g0g4_impulse_clean\residual_ensemble_v23.pt ^
  --world-model-aleatoric-scale 1.5 ^
  --noise-era 10hz ^
  --multigate-vision ^
  --spawn-at-rest ^
  --impulse-rate-hz 0.08 ^
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
  --finish-time-target-s 8.7 ^
  --finish-time-bonus-per-s 200.0 ^
  --gate-time-targets 2.33 1.80 1.00 1.35 2.20 ^
  --gate-time-bonus-per-s 200.0 ^
  --demo-crossing-bonus 2.0 ^
  --demo-crossing-radius-m 0.50 ^
  --eval-interval 5 ^
  --eval-envs 2048 ^
  --eval-seed 20260868 ^
  --eval-impulse-rate-hz 0.08 ^
  1>D:\ai-gp\worldmodel\ppo_g0g4_v22v23_tier87_v1.stdout.log ^
  2>D:\ai-gp\worldmodel\ppo_g0g4_v22v23_tier87_v1.stderr.log
exit /b %ERRORLEVEL%

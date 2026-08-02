@echo off
cd /d C:\Users\henry\aigp
.venv-train\Scripts\python.exe -u scripts\fastsim_train_ppo.py ^
  --eval-only ^
  --n-envs 2048 ^
  --eval-envs 2048 ^
  --eval-seed 20260805 ^
  --run-dir worldmodel\ppo_multimodel_longcredit_v7\gate4_authority_audit ^
  --resume worldmodel\ppo_multimodel_longcredit_v7\best.pt ^
  --rate-sign 1 ^
  --model data\fastsim_model_v2.json ^
  --map data\vq2_runtime_map_g9g15fix.json ^
  --demo-npz data\vq2_g0g1fast9267_g2plus_clean_demo.npz ^
  --bc-init data\vq2_g0g1fast9267_g2plus_clean_demo.npz ^
  --residual ^
  --residual-scale 0.20 ^
  --live-teacher-config worldmodel\teacher_full17_cem_safe_v1.json ^
  --controller-model data\fastsim_model_v2.json ^
  --race-gates 5 ^
  --max-episode-s 12 ^
  --random-start-frac 0 ^
  --world-model ^
    worldmodel\multimodel_seqg2_probe\residual_ensemble_current_v13.pt ^
    worldmodel\multimodel_seqg2_probe\residual_ensemble_v19.pt ^
    worldmodel\v21_full16_baseline\residual_ensemble_v20.pt ^
    worldmodel\v21_full16_baseline\residual_ensemble_v21.pt ^
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
  --eval-scale-profiles "1,1,1,1,1;1,1,1,1,1.5;1,1,1,1,2;1,1,1,1,3"

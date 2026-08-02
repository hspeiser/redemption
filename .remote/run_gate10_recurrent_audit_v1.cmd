@echo off
setlocal
cd /d C:\Users\henry\aigp

.venv-train\Scripts\python.exe -u scripts\audit_vq2_recurrent_gate_actor.py ^
  --candidate worldmodel\flywheel_gate10_v1\candidate.pt ^
  --normalization-checkpoint data\flywheel\primary_ppo_v8.pt ^
  --ensemble worldmodel\v24_allgate_liveab\residual_ensemble_v24.pt ^
  --ensemble worldmodel\v25_allgate_v8\residual_ensemble_v25.pt ^
  --model data\fastsim_model_v2.json ^
  --controller-model data\fastsim_model_v2.json ^
  --demo data\vq2_g0g1fast9267_g2plus_clean_demo.npz ^
  --map data\vq2_runtime_map_g9g15fix.json ^
  --obstacles data\vq2_obstacles_inflated.json ^
  --teacher-config worldmodel\teacher_full17_cem_safe_v1.json ^
  --out worldmodel\flywheel_gate10_v1\fresh_audit.json ^
  --focus-gate 10 ^
  --worlds 2048 ^
  --seed 20260802101 ^
  --max-episode-s 32 ^
  --aleatoric-scale 1.5 ^
  --impulse-rate-hz 0.08 ^
  --residual-scale 0.20 ^
  --multigate-estimator ^
  --device cuda ^
  > worldmodel\flywheel_gate10_v1\audit.stdout.log ^
  2> worldmodel\flywheel_gate10_v1\audit.stderr.log

echo %ERRORLEVEL% > worldmodel\flywheel_gate10_v1\audit.exit_code.txt

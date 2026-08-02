@echo off
setlocal
cd /d C:\Users\henry\aigp

.venv-train\Scripts\python.exe -u scripts\train_vq2_recurrent_iql.py ^
  --dataset C:\Users\henry\aigp\data\flywheel\gate5_real_v2 ^
  --seed-checkpoint C:\Users\henry\aigp\data\flywheel\v79_seed.pt ^
  --normalization-checkpoint C:\Users\henry\aigp\data\flywheel\primary_ppo_v8.pt ^
  --output C:\Users\henry\aigp\worldmodel\flywheel_gate5_v1\candidate.pt ^
  --focus-gate 5 ^
  --critic-steps 20000 ^
  --bc-steps 5000 ^
  --awr-steps 5000 ^
  --batch 1024 ^
  --batch-sequences 24 ^
  --critic-lr 0.0003 ^
  --actor-lr 0.0002 ^
  --expectile 0.7 ^
  --temperature 3.0 ^
  --device cuda ^
  > C:\Users\henry\aigp\worldmodel\flywheel_gate5_v1\stdout.log ^
  2> C:\Users\henry\aigp\worldmodel\flywheel_gate5_v1\stderr.log

echo %ERRORLEVEL% > C:\Users\henry\aigp\worldmodel\flywheel_gate5_v1\exit_code.txt

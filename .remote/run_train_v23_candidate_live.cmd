@echo off
cd /d C:\Users\henry\aigp
.venv-train\Scripts\python.exe -u scripts\train_vq2_g0g4_worldmodel.py ^
  --dataset worldmodel\g0g4_current_aug_18_candidate_v5_live ^
  --base-model data\fastsim_model_v2.json ^
  --map data\vq2_runtime_map_g9g15fix.json ^
  --out worldmodel\v23_candidate_live\residual_ensemble_v23.pt ^
  --device cuda ^
  --seed 20260877 ^
  --members 5 ^
  --epochs 40 ^
  --batch-size 2048 ^
  --lr 0.0002 ^
  --rollout-finetune-epochs 2 ^
  --rollout-horizons 8,16,32 ^
  --rollout-lr 0.00005 ^
  --rollout-batch-size 256 ^
  --rollout-max-starts 5000 ^
  --resume-checkpoint worldmodel\v23_candidate_live\residual_ensemble_v23.partial.pt

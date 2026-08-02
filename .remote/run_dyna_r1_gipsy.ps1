$ErrorActionPreference = 'Stop'
$root = 'C:\Users\henry\aigp_dyna_r1'
$python = 'C:\Users\henry\aigp\.venv-train\Scripts\python.exe'
Set-Location -LiteralPath $root
& $python -u scripts\train_vq2_g0g4_worldmodel.py `
    --dataset worldmodel\g0g4_current_aug_20_live_abba_v7 `
    --base-model data\fastsim_model_v2.json `
    --map data\vq2_runtime_map_g9g15fix.json `
    --init-ensemble worldmodel\v25_live_abba_v7\residual_ensemble_v24.pt `
    --out worldmodel\v25_live_abba_v7\residual_ensemble_v25.pt `
    --members 5 `
    --epochs 30 `
    --batch-size 4096 `
    --lr 0.00008 `
    --rollout-finetune-epochs 3 `
    --rollout-horizons 12,24,32 `
    --rollout-lr 0.000015 `
    --rollout-batch-size 1024 `
    --rollout-max-starts 12288 `
    --seed 20260892 `
    --device cuda
exit $LASTEXITCODE

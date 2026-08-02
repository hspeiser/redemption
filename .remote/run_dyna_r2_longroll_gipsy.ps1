$ErrorActionPreference = 'Stop'
$root = 'C:\Users\henry\aigp_dyna_r1'
$python = 'C:\Users\henry\aigp\.venv-train\Scripts\python.exe'
Set-Location -LiteralPath $root
& $python -u scripts\train_vq2_g0g4_worldmodel.py `
    --dataset worldmodel\g0g4_current_aug_21_live_abba_v7_longroll `
    --base-model data\fastsim_model_v2.json `
    --map data\vq2_runtime_map_g9g15fix.json `
    --init-ensemble worldmodel\v25_live_abba_v7\residual_ensemble_v25.pt `
    --skip-one-step-finetune `
    --out worldmodel\v26_live_abba_v7_longroll\residual_ensemble_v26.pt `
    --members 5 `
    --rollout-finetune-epochs 2 `
    --rollout-horizons 48,64,96 `
    --rollout-lr 0.000006 `
    --rollout-batch-size 512 `
    --rollout-max-starts 4096 `
    --seed 20260897 `
    --device cuda
exit $LASTEXITCODE

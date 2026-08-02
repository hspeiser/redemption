$repo = 'C:\Users\henry\aigp_dyna_r1'
$python = 'C:\Users\henry\aigp\.venv-train\Scripts\python.exe'
$outDir = Join-Path $repo 'worldmodel\v27_recent_abba_holdout'
$stdout = Join-Path $outDir 'stdout.log'
$stderr = Join-Path $outDir 'stderr.log'
$outModel = Join-Path $outDir 'residual_ensemble_v27_holdout.pt'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Set-Location $repo
& $python -u scripts\train_vq2_g0g4_worldmodel.py `
    --dataset worldmodel\g0g4_current_aug_22_recent_abba_holdout `
    --base-model data\fastsim_model_v2.json `
    --map data\vq2_runtime_map_g9g15fix.json `
    --out $outModel `
    --device cuda --seed 20261008 --members 5 `
    --epochs 2 --batch-size 1024 --lr 0.0001 `
    --rollout-finetune-epochs 1 `
    --rollout-horizons 48,64,96 `
    --rollout-lr 0.00003 --rollout-batch-size 128 `
    --rollout-max-starts 8192 `
    --init-ensemble worldmodel\v26_live_abba_v7_longroll\residual_ensemble_v26.pt `
    1> $stdout 2> $stderr
$LASTEXITCODE | Set-Content -LiteralPath (Join-Path $outDir 'exitcode.txt')

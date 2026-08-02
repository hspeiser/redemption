$repo = 'C:\Users\henry\Desktop\ai-gp'
$python = Join-Path $repo '.venv-train\Scripts\python.exe'
$outDir = 'D:\ai-gp\worldmodel\v27_recent_abba'
$outModel = Join-Path $outDir 'residual_ensemble_v27.pt'
$stdout = 'D:\ai-gp\worldmodel\v27_recent_abba.stdout.log'
$stderr = 'D:\ai-gp\worldmodel\v27_recent_abba.stderr.log'
$arguments = @(
    '-u', 'scripts\train_vq2_g0g4_worldmodel.py',
    '--dataset', 'D:\ai-gp\worldmodel\g0g4_current_aug_22_recent_abba',
    '--base-model', 'data\fastsim_model_v2.json',
    '--map', 'data\vq2_runtime_map_g9g15fix.json',
    '--out', $outModel,
    '--device', 'cuda', '--seed', '20261007', '--members', '5',
    '--epochs', '2', '--batch-size', '1024', '--lr', '0.0001',
    '--rollout-finetune-epochs', '1',
    '--rollout-horizons', '48,64,96',
    '--rollout-lr', '0.00003', '--rollout-batch-size', '128',
    '--rollout-max-starts', '8192',
    '--init-ensemble', 'D:\ai-gp\worldmodel\v26_live_abba_v7_longroll\residual_ensemble_v26.pt'
)
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$process = Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $repo `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru
Write-Output $process.Id

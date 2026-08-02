$repo = 'C:\Users\henry\Desktop\ai-gp'
$python = Join-Path $repo '.venv-train\Scripts\python.exe'
$outDir = 'D:\ai-gp\worldmodel\v27_micro_audits'
$ensemble = 'D:\ai-gp\worldmodel\v27_recent_abba\residual_ensemble_v27.pt'
$dataset = 'D:\ai-gp\worldmodel\g0g4_current_aug_22_recent_abba'
$candidates = @(
    'baseline',
    'g3t103',
    'g3t106',
    'g4t080',
    'g4t082',
    'g3t103_g4t082'
)
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Set-Location $repo
$env:PYTHONPATH = $repo
foreach ($name in $candidates) {
    $candidate = "D:\ai-gp\worldmodel\g0g4_v27_micro_$name.json"
    $out = Join-Path $outDir "$name.json"
    $stdout = Join-Path $outDir "$name.stdout.log"
    $stderr = Join-Path $outDir "$name.stderr.log"
    & $python -u scripts\audit_vq2_g0g4_candidate.py `
        --candidate $candidate `
        --dataset $dataset `
        --ensemble $ensemble `
        --model data\fastsim_model_v2.json `
        --controller-model data\fastsim_model_v2.json `
        --actor worldmodel\ppo_multimodel_segmentcredit_v8\best.pt `
        --demo data\vq2_g0g1fast9267_g2plus_clean_demo.npz `
        --map data\vq2_runtime_map_g9g15fix.json `
        --obstacles data\vq2_obstacles_inflated.json `
        --teacher-config 'D:\ai-gp\worldmodel\g0g4_v22_g3center_live_config_v1.json' `
        --worlds 4096 --seed 20261009 --device cuda `
        --aleatoric-scale 1.5 --impulse-rate-hz 0.08 `
        --residual-scale 0.20 --geometry-limit 0.30 `
        --tier-target 8.7 --multigate-vision --live-estimator-realism `
        --out $out 1> $stdout 2> $stderr
    if ($LASTEXITCODE -ne 0) {
        throw "audit $name failed with exit code $LASTEXITCODE"
    }
}

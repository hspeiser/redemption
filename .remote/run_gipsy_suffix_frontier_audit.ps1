$ErrorActionPreference = "Stop"

$repo = "C:\Users\henry\aigp"
$outDir = Join-Path $repo "worldmodel\suffix_exact_straight_lead_v1"
$python = Join-Path $repo ".venv-train\Scripts\python.exe"
$evaluator = Join-Path $repo "scripts\liveteacher_layer2.py"
$geometryConfig = Join-Path $repo "worldmodel\gipsy_geometry_config.json"
$baselineConfig = Join-Path $repo "worldmodel\gipsy_config_search.json"
$fastConfig = Join-Path $outDir "speed_config.json"

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

function New-CandidateConfig {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$VelocityScales,
        [Parameter(Mandatory = $true)][string]$ActionLeads
    )

    $config = Get-Content -Raw -LiteralPath $geometryConfig | ConvertFrom-Json
    $config.args.reference_velocity_scales = $VelocityScales
    $config.args.reference_action_leads = $ActionLeads
    $json = $config | ConvertTo-Json -Depth 100
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $json, $utf8NoBom)
}

$priorConfig = Join-Path $outDir "prior_safe_config.json"
New-CandidateConfig -Path $priorConfig `
    -VelocityScales "0:1.53180218,1:1.82969701,2:1.49190915,3:1.31209302,4:2.20000005,11:0.999634797,12:1.08475459,13:1.05,14:1,15:1.10064872,16:1.1546934" `
    -ActionLeads "0:1,1:4,2:14,3:1,4:2"

$reliableConfig = Join-Path $outDir "reliability_knee_config.json"
New-CandidateConfig -Path $reliableConfig `
    -VelocityScales "0:1.53180218,1:1.82969701,2:1.49190915,3:1.31209302,4:2.20000005,11:1.014624055,12:1.031248149,13:1.05,14:1,15:1.062275906,16:1.198293646" `
    -ActionLeads "0:1,1:4,2:14,3:1,4:2,11:2,12:1,13:0,14:0,15:0,16:2"

$common = @(
    $evaluator,
    "--model", (Join-Path $repo "data\fastsim_model_v3_live.json"),
    "--ensemble",
    (Join-Path $repo "worldmodel\v25_allgate_v8\residual_ensemble_v25.pt"),
    (Join-Path $repo "worldmodel\v28_allgate_registry_flywheel\residual_ensemble_v28.pt"),
    (Join-Path $repo "worldmodel\v30_allgate_registry_flywheel\residual_ensemble_v30.pt"),
    "--worlds", "768",
    "--seed", "20261306",
    "--speed-cap", "12.5",
    "--aleatoric-scale", "1.5",
    "--max-episode-s", "20",
    "--demo", (Join-Path $repo "data\vq2_hybrid_fastprefix_a2suffix_g11_v2.npz"),
    "--handoff-pool", (Join-Path $repo "data\lineopt\handoff_pool_13finish.json"),
    "--device", "cuda",
    "--controller", "batched"
)

$arms = @(
    @{ Name = "baseline"; Config = $baselineConfig },
    @{ Name = "prior_safe"; Config = $priorConfig },
    @{ Name = "fast"; Config = $fastConfig },
    @{ Name = "reliable"; Config = $reliableConfig }
)

$processes = @()
foreach ($arm in $arms) {
    $out = Join-Path $outDir ("paired_20261306_{0}.npz" -f $arm.Name)
    $stdout = Join-Path $outDir ("paired_20261306_{0}.log" -f $arm.Name)
    $stderr = Join-Path $outDir ("paired_20261306_{0}.err.log" -f $arm.Name)
    $arguments = @($common + @("--config", $arm.Config, "--out", $out))
    $processes += Start-Process -FilePath $python -ArgumentList $arguments `
        -WorkingDirectory $repo -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
    Write-Host ("started {0} pid={1}" -f $arm.Name, $processes[-1].Id)
}

$processes | Wait-Process

$failed = $false
for ($i = 0; $i -lt $arms.Count; $i++) {
    $processes[$i].Refresh()
    Write-Host ("finished {0} exit={1}" -f $arms[$i].Name, $processes[$i].ExitCode)
    if ($processes[$i].ExitCode -ne 0) {
        $failed = $true
        Get-Content -LiteralPath (Join-Path $outDir ("paired_20261306_{0}.err.log" -f $arms[$i].Name))
    }
}

if ($failed) { exit 1 }

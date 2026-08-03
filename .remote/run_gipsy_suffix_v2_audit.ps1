$ErrorActionPreference = "Stop"

$repo = "C:\Users\henry\aigp"
$outDir = Join-Path $repo "worldmodel\suffix_exact_straight_lead_v2"
$python = Join-Path $repo ".venv-train\Scripts\python.exe"
$evaluator = Join-Path $repo "scripts\liveteacher_layer2.py"
$common = @(
    $evaluator,
    "--model", (Join-Path $repo "data\fastsim_model_v3_live.json"),
    "--ensemble",
    (Join-Path $repo "worldmodel\v25_allgate_v8\residual_ensemble_v25.pt"),
    (Join-Path $repo "worldmodel\v28_allgate_registry_flywheel\residual_ensemble_v28.pt"),
    (Join-Path $repo "worldmodel\v30_allgate_registry_flywheel\residual_ensemble_v30.pt"),
    "--worlds", "768",
    "--seed", "20261321",
    "--speed-cap", "12.5",
    "--aleatoric-scale", "1.5",
    "--max-episode-s", "20",
    "--demo", (Join-Path $repo "data\vq2_hybrid_fastprefix_a2suffix_g11_v2.npz"),
    "--handoff-pool", (Join-Path $repo "data\lineopt\handoff_pool_13finish.json"),
    "--device", "cuda",
    "--controller", "batched"
)

$arms = @(
    @{ Name = "baseline"; Config = (Join-Path $repo "worldmodel\gipsy_config_search.json") },
    @{ Name = "v1_fast"; Config = (Join-Path $repo "worldmodel\suffix_exact_straight_lead_v1\speed_config.json") },
    @{ Name = "v1_reliable"; Config = (Join-Path $repo "worldmodel\suffix_exact_straight_lead_v1\reliability_knee_config.json") },
    @{ Name = "v2_balanced"; Config = (Join-Path $outDir "speed_config.json") }
)

$processes = @()
foreach ($arm in $arms) {
    $out = Join-Path $outDir ("paired_20261321_{0}.npz" -f $arm.Name)
    $stdout = Join-Path $outDir ("paired_20261321_{0}.log" -f $arm.Name)
    $stderr = Join-Path $outDir ("paired_20261321_{0}.err.log" -f $arm.Name)
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
        Get-Content -LiteralPath (Join-Path $outDir ("paired_20261321_{0}.err.log" -f $arms[$i].Name))
    }
}
if ($failed) { exit 1 }

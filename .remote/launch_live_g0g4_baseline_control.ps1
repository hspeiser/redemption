param(
    [int]$Episodes = 1,
    [string]$OutputRoot = 'D:\ai-gp\training\vq2_g0g4_ab_v1_baseline',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\henry\Desktop\ai-gp'
$python = Join-Path $repo '.venv-train\Scripts\python.exe'
$launcher = Join-Path $repo 'scripts\launch_vq2_from_config.py'
$sourceConfig = 'D:\ai-gp\training\vq2_g0g4_impulse_id_v1\20260802_001532\config.json'
$recordRoot = 'D:\ai-gp\raw_sessions'
$map = Join-Path $repo 'data\vq2_runtime_map_g9g15fix.json'
$primary = Join-Path $repo 'data\models\gatenet_v7_best.pt'
$logRoot = 'D:\ai-gp\runlogs'

foreach ($required in @($python, $launcher, $sourceConfig, $map, $primary)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required file not found: $required"
    }
}

New-Item -ItemType Directory -Force -Path $OutputRoot, $recordRoot, $logRoot | Out-Null
$env:AIGP_MULTIGATE = '1'
$arguments = @(
    '-u', $launcher,
    '--config', $sourceConfig,
    '--output-root', $OutputRoot,
    '--episodes', "$Episodes",
    '--eval-only',
    '--poc-stop-after-gate', '4',
    '--override', "map=$map",
    '--override', "primary=$primary",
    '--override', "record_root=$recordRoot",
    '--override', 'full_recording=true',
    '--override', 'vision_device=cuda',
    '--override', 'vision_hz=10.0',
    '--override', 'crop_tracker=true',
    '--override', 'crop_tracker_hz=10.0',
    '--override', 'official_countdown=true',
    '--override', 'domain_impulse_probability=0.0',
    '--override', 'domain_impulse_amplitude=0.0',
    '--override', 'max_episode_sim_step_p95=0.055',
    '--override', 'max_episode_sim_step_max=0.30',
    '--override', 'max_episode_step_p95_ms=60.0',
    '--override', 'timing_failure_limit=1'
)

if ($DryRun) {
    [pscustomobject]@{
        executable = $python
        arguments = $arguments
        multigate = $true
        arm = 'baseline'
        output_root = $OutputRoot
    } | ConvertTo-Json -Compress
    exit 0
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logRoot "vq2_g0g4_ab_baseline_$stamp.stdout.log"
$stderr = Join-Path $logRoot "vq2_g0g4_ab_baseline_$stamp.stderr.log"
$process = Start-Process -FilePath $python -ArgumentList $arguments `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr
[pscustomobject]@{
    process_id = $process.Id
    stdout = $stdout
    stderr = $stderr
    output_root = $OutputRoot
    arm = 'baseline'
} | ConvertTo-Json -Compress

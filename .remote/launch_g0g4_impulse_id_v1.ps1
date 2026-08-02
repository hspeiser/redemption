param(
    [int]$Episodes = 32,
    [double]$ImpulseProbability = 0.75,
    [double]$ImpulseAmplitude = 0.025,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\henry\Desktop\ai-gp'
$python = Join-Path $repo '.venv-train\Scripts\python.exe'
$launcher = Join-Path $repo 'scripts\launch_vq2_from_config.py'
$sourceConfig = 'D:\ai-gp\training\vq2_v93_v79_gate10_macro_awr\20260731_220141\config.json'
$outputRoot = 'D:\ai-gp\training\vq2_g0g4_impulse_id_v1'
$recordRoot = 'D:\ai-gp\raw_sessions'
$map = Join-Path $repo 'data\vq2_runtime_map_g9g15fix.json'
$primary = Join-Path $repo 'data\models\gatenet_v7_best.pt'
$logRoot = 'D:\ai-gp\runlogs'

foreach ($required in @($python, $launcher, $sourceConfig, $map, $primary)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required file not found: $required"
    }
}

New-Item -ItemType Directory -Force -Path $outputRoot, $recordRoot, $logRoot | Out-Null
$env:AIGP_MULTIGATE = '1'
$arguments = @(
    '-u', $launcher,
    '--config', $sourceConfig,
    '--output-root', $outputRoot,
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
    '--override', 'train_gate=-1',
    '--override', 'official_countdown=true',
    '--override', 'max_transition_sim_time=0.5',
    '--override', 'max_episode_sim_step_p95=0.055',
    '--override', 'max_episode_sim_step_max=0.30',
    '--override', 'max_episode_step_p95_ms=60.0',
    '--override', 'timing_failure_limit=2',
    '--override', "domain_impulse_probability=$ImpulseProbability",
    '--override', 'domain_impulse_gates=0,1,2,3',
    '--override', 'domain_impulse_axes=0,1,3',
    '--override', "domain_impulse_amplitude=$ImpulseAmplitude",
    '--override', 'domain_impulse_duration_steps=4',
    '--override', 'domain_impulse_min_distance=4.0',
    '--override', 'domain_impulse_max_distance=8.0'
)

if ($DryRun) {
    [pscustomobject]@{
        executable = $python
        arguments = $arguments
        multigate = $true
        purpose = 'protected g0-g4 structured impulse identification'
    } | ConvertTo-Json -Depth 5
    exit 0
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logRoot "vq2_g0g4_impulse_id_$stamp.stdout.log"
$stderr = Join-Path $logRoot "vq2_g0g4_impulse_id_$stamp.stderr.log"
$process = Start-Process -FilePath $python -ArgumentList $arguments `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr
[pscustomobject]@{
    process_id = $process.Id
    stdout = $stdout
    stderr = $stderr
}

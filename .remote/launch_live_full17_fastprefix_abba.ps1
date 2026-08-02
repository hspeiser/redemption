param(
    [int]$Cycles = 1,
    [string]$CandidateConfig = 'D:\ai-gp\worldmodel\g0g4_v22_g3center_live_config_v1.json',
    [string]$ChampionConfig = 'D:\ai-gp\training\vq2_fullcourse_reliablebackbone_all17_v1\20260801_160352\config.json',
    [string]$Checkpoint = 'C:\Users\henry\Desktop\ai-gp\worldmodel\ppo_multimodel_segmentcredit_v8\best.pt',
    [string]$Primary = 'C:\Users\henry\Desktop\ai-gp\data\models\gatenet_v7_best.pt',
    [string]$PrefixPrimary = 'C:\Users\henry\Desktop\ai-gp\data\models\gatenet_v13drought_best.pt',
    [string]$PrefixPrimaryGates = '0,1,2,3,4',
    [string]$ResidualGates = '0,1,2,4',
    [string]$Sequence = 'protected_champion,candidate,candidate,protected_champion',
    [double]$OfficialReleaseMarginMs = 20.0,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\henry\Desktop\ai-gp'
$python = Join-Path $repo '.venv-train\Scripts\python.exe'
$launcher = Join-Path $repo 'scripts\launch_vq2_from_config.py'
$outputRoot = 'D:\ai-gp\training\vq2_full17_fastprefix_abba_v1'
$recordRoot = 'D:\ai-gp\raw_sessions'
$map = Join-Path $repo 'data\vq2_runtime_map_g9g15fix.json'
$logRoot = 'D:\ai-gp\runlogs'
$probeSequence = @(
    $Sequence.Split(',') | ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
)
if (-not $probeSequence.Count) {
    throw 'Sequence must contain at least one probe arm'
}
$invalidArms = @($probeSequence | Where-Object {
    $_ -notin @('protected_champion', 'candidate')
})
if ($invalidArms.Count) {
    throw "Invalid probe arm(s): $($invalidArms -join ',')"
}
$episodes = $probeSequence.Count * [Math]::Max(1, $Cycles)
$sequenceText = $probeSequence -join ','

foreach ($required in @(
    $python, $launcher, $CandidateConfig, $ChampionConfig, $Checkpoint,
    $map, $Primary, $PrefixPrimary
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required file not found: $required"
    }
}

New-Item -ItemType Directory -Force -Path `
    $outputRoot, $recordRoot, $logRoot | Out-Null
$env:AIGP_MULTIGATE = '1'
$arguments = @(
    '-u', $launcher,
    '--config', $CandidateConfig,
    '--output-root', $outputRoot,
    '--episodes', "$episodes",
    '--eval-only',
    '--override', 'poc_stop_after_gate=-1',
    '--override', "ppo_residual_checkpoint=$Checkpoint",
    '--override', "residual_gates=$ResidualGates",
    '--override', 'train_gate=-1',
    '--override', 'residual_scale=0.20',
    '--override', 'interleave_protected_champion=true',
    '--override', "interleave_champion_config=$ChampionConfig",
    '--override', "probe_arm_sequence=$sequenceText",
    '--override', "map=$map",
    '--override', "primary=$Primary",
    '--override', "gate_primary=$PrefixPrimary",
    '--override', "gate_primary_gates=$PrefixPrimaryGates",
    '--override', "record_root=$recordRoot",
    '--override', 'full_recording=true',
    '--override', 'vision_device=cuda',
    '--override', 'vision_hz=10.0',
    '--override', 'crop_tracker=true',
    '--override', 'crop_tracker_hz=10.0',
    '--override', 'official_countdown=true',
    '--override', "official_release_margin_ms=$OfficialReleaseMarginMs",
    '--override', 'right_lateral_action_bias=0.0',
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
        sequence = $probeSequence
        episodes = $episodes
        candidate_config = (Resolve-Path -LiteralPath $CandidateConfig).Path
        champion_config = (Resolve-Path -LiteralPath $ChampionConfig).Path
        checkpoint = (Resolve-Path -LiteralPath $Checkpoint).Path
        primary = (Resolve-Path -LiteralPath $Primary).Path
        prefix_primary = (Resolve-Path -LiteralPath $PrefixPrimary).Path
        prefix_primary_gates = $PrefixPrimaryGates
        residual_gates = $ResidualGates
        one_process = $true
        simulator_restart = $false
        strict_timing_abort = $true
    } | ConvertTo-Json -Depth 6
    exit 0
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logRoot "vq2_full17_fastprefix_abba_$stamp.stdout.log"
$stderr = Join-Path $logRoot "vq2_full17_fastprefix_abba_$stamp.stderr.log"
$process = Start-Process -FilePath $python -ArgumentList $arguments `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr
[pscustomobject]@{
    process_id = $process.Id
    stdout = $stdout
    stderr = $stderr
    output_root = $outputRoot
    sequence = $probeSequence
    episodes = $episodes
    one_process = $true
} | ConvertTo-Json -Compress

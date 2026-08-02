param(
    [int]$Cycles = 1,
    [string]$CandidateConfig = 'D:\ai-gp\worldmodel\g0g4_v22_g3center_live_config_v1.json',
    [string]$ChampionConfig = 'D:\ai-gp\champions\vq2_36s_20260802\training_session\config.json',
    [bool]$ChampionUsesResidual = $true,
    [string]$Checkpoint = 'C:\Users\henry\Desktop\ai-gp\worldmodel\ppo_multimodel_segmentcredit_v8\best.pt',
    [string]$SecondaryCheckpoint = 'C:\Users\henry\Desktop\ai-gp\worldmodel\ppo_all17_multimodel_v3_late_safe\best.pt',
    [string]$SecondaryResidualGates = '11,12,14,15,16',
    [string]$Primary = 'C:\Users\henry\Desktop\ai-gp\data\models\gatenet_v7_best.pt',
    [string]$PrefixPrimary = 'C:\Users\henry\Desktop\ai-gp\data\models\gatenet_v13drought_best.pt',
    [string]$PrefixPrimaryGates = '3,4,5',
    [string]$ResidualGates = '0,1,2,4,11,12,14,15,16',
    [string]$ReferenceLateralOffsets = '0:0,1:0.300000012,2:0.0878505111,3:-0.3,4:0.45,5:0.10,6:-0.10',
    [string]$GateCenterFunnelGates = '3,5,8',
    [string]$Sequence = 'protected_champion,candidate,candidate,protected_champion',
    [ValidateRange(1, 10)]
    [int]$TimingFailureLimit = 2,
    # Live-validated on 2026-08-02: -500 ms was rejected, while a measured
    # -142 ms release completed and was accepted by the qualifier UI.
    [double]$OfficialReleaseMarginMs = -150.0,
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
    $SecondaryCheckpoint,
    $map, $Primary, $PrefixPrimary
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required file not found: $required"
    }
}

New-Item -ItemType Directory -Force -Path `
    $outputRoot, $recordRoot, $logRoot | Out-Null
$env:AIGP_MULTIGATE = '1'
$env:AIGP_GATE_PRIMARY_ENSEMBLE = '1'
$arguments = @(
    '-u', $launcher,
    '--config', $CandidateConfig,
    '--output-root', $outputRoot,
    '--episodes', "$episodes",
    '--eval-only',
    '--override', 'poc_stop_after_gate=-1',
    '--override', "ppo_residual_checkpoint=$Checkpoint",
    '--override', "secondary_ppo_residual_checkpoint=$SecondaryCheckpoint",
    '--override', "secondary_ppo_residual_gates=$SecondaryResidualGates",
    '--override', "residual_gates=$ResidualGates",
    '--override', "reference_lateral_offsets=$ReferenceLateralOffsets",
    '--override', "gate_center_funnel_gates=$GateCenterFunnelGates",
    '--override', 'train_gate=-1',
    '--override', 'residual_scale=0.20',
    '--override', 'interleave_protected_champion=true',
    '--override', "interleave_champion_config=$ChampionConfig",
    '--override', "interleave_champion_residual=$ChampionUsesResidual",
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
    '--override', "timing_failure_limit=$TimingFailureLimit"
)

if ($DryRun) {
    [pscustomobject]@{
        executable = $python
        arguments = $arguments
        sequence = $probeSequence
        episodes = $episodes
        candidate_config = (Resolve-Path -LiteralPath $CandidateConfig).Path
        champion_config = (Resolve-Path -LiteralPath $ChampionConfig).Path
        champion_uses_residual = $ChampionUsesResidual
        checkpoint = (Resolve-Path -LiteralPath $Checkpoint).Path
        secondary_checkpoint = (
            Resolve-Path -LiteralPath $SecondaryCheckpoint
        ).Path
        secondary_residual_gates = $SecondaryResidualGates
        primary = (Resolve-Path -LiteralPath $Primary).Path
        prefix_primary = (Resolve-Path -LiteralPath $PrefixPrimary).Path
        prefix_primary_gates = $PrefixPrimaryGates
        residual_gates = $ResidualGates
        reference_lateral_offsets = $ReferenceLateralOffsets
        gate_center_funnel_gates = $GateCenterFunnelGates
        gate_primary_ensemble = $true
        one_process = $true
        simulator_restart = $false
        strict_timing_abort = $true
        timing_failure_limit = $TimingFailureLimit
    } | ConvertTo-Json -Depth 6
    exit 0
}

$pipelineDashboard = Join-Path $repo '.remote\launch_vq2_pipeline_dashboard.ps1'
& powershell -NoProfile -ExecutionPolicy Bypass -File $pipelineDashboard |
    Out-Null

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

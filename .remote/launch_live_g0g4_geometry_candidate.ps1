param(
    [Parameter(Mandatory = $true)]
    [string]$Candidate,
    [int]$Episodes = 4,
    [string]$Checkpoint = 'C:\Users\henry\Desktop\ai-gp\worldmodel\ppo_multimodel_segmentcredit_v8\best.pt',
    [string]$Primary = 'C:\Users\henry\Desktop\ai-gp\data\models\gatenet_v7_best.pt',
    [switch]$Multigate,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\henry\Desktop\ai-gp'
$python = Join-Path $repo '.venv-train\Scripts\python.exe'
$launcher = Join-Path $repo 'scripts\launch_vq2_from_config.py'
$config = 'D:\ai-gp\worldmodel\teacher_full17_cem_safe_v1.json'
$outputRoot = 'D:\ai-gp\training\vq2_g0g4_geometry_live'
$recordRoot = 'D:\ai-gp\raw_sessions'
$map = Join-Path $repo 'data\vq2_runtime_map_g9g15fix.json'
$logRoot = 'D:\ai-gp\runlogs'

foreach ($required in @($Candidate, $Checkpoint, $Primary, $config, $map)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required file not found: $required"
    }
}

$payload = Get-Content -LiteralPath $Candidate -Raw | ConvertFrom-Json
$requiredOverrides = @(
    'reference_action_leads',
    'reference_thrust_scales',
    'reference_velocity_scales',
    'reference_rate_scales',
    'trajectory_blends',
    'reference_lateral_offsets',
    'reference_vertical_offsets'
)
$candidateOverrides = [ordered]@{}
if ($null -ne $payload.live_overrides) {
    foreach ($name in $requiredOverrides) {
        if ($payload.live_overrides.PSObject.Properties.Name -notcontains $name) {
            throw "Candidate live_overrides is missing '$name'"
        }
        $candidateOverrides[$name] = [string]$payload.live_overrides.$name
    }
} else {
    $arrays = [ordered]@{
        reference_action_leads = @($payload.leads)
        reference_thrust_scales = @($payload.thrust_scales)
        reference_velocity_scales = @($payload.velocity_scales)
        reference_rate_scales = @($payload.rate_scales)
        trajectory_blends = @($payload.trajectory_blends)
        reference_lateral_offsets = @($payload.lateral_offsets_m)
        reference_vertical_offsets = @($payload.vertical_offsets_m)
    }
    foreach ($name in $requiredOverrides) {
        $values = @($arrays[$name])
        if ($values.Count -eq 0 -or $null -eq $values[0]) {
            if ($name -match 'offsets$') {
                $values = @(0.0, 0.0, 0.0, 0.0, 0.0)
            } elseif ($name -eq 'reference_rate_scales') {
                $values = @(1.0, 1.0, 1.0, 1.0, 1.0)
            } else {
                throw "Candidate is missing values for '$name'"
            }
        }
        if ($values.Count -ne 5) {
            throw "Candidate '$name' must contain 5 gate values"
        }
        $pairs = for ($gate = 0; $gate -lt 5; $gate++) {
            $value = if ($name -eq 'reference_action_leads') {
                [int]$values[$gate]
            } else {
                [double]$values[$gate]
            }
            "${gate}:$value"
        }
        $candidateOverrides[$name] = $pairs -join ','
    }
}

New-Item -ItemType Directory -Force -Path $outputRoot, $logRoot | Out-Null
$env:AIGP_MULTIGATE = if ($Multigate) { '1' } else { '0' }

$arguments = @(
    '-u', $launcher,
    '--config', $config,
    '--output-root', $outputRoot,
    '--episodes', "$Episodes",
    '--eval-only',
    '--poc-stop-after-gate', '4',
    '--override', "ppo_residual_checkpoint=$Checkpoint",
    '--override', 'residual_gates=0,1,2,3,4',
    '--override', 'train_gate=-1',
    '--override', 'residual_scale=0.20',
    '--override', 'interleave_protected_champion=false',
    '--override', "map=$map",
    '--override', "primary=$Primary",
    '--override', "record_root=$recordRoot",
    '--override', 'full_recording=true',
    '--override', 'vision_device=cuda',
    '--override', 'vision_hz=10.0',
    '--override', 'crop_tracker=true',
    '--override', 'crop_tracker_hz=10.0',
    '--override', 'official_countdown=true',
    '--override', 'right_lateral_action_bias=0.0'
    '--override', 'max_episode_sim_step_p95=0.055',
    '--override', 'max_episode_sim_step_max=0.30',
    '--override', 'max_episode_step_p95_ms=60.0',
    '--override', 'timing_failure_limit=2'
)
foreach ($name in $requiredOverrides) {
    $value = $candidateOverrides[$name]
    $arguments += @('--override', "$name=$value")
}

if ($DryRun) {
    [pscustomobject]@{
        executable = $python
        arguments = $arguments
        multigate = [bool]$Multigate
        candidate = (Resolve-Path -LiteralPath $Candidate).Path
        checkpoint = (Resolve-Path -LiteralPath $Checkpoint).Path
    } | ConvertTo-Json -Depth 5
    exit 0
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logRoot "vq2_g0g4_geometry_$stamp.stdout.log"
$stderr = Join-Path $logRoot "vq2_g0g4_geometry_$stamp.stderr.log"
$process = Start-Process -FilePath $python -ArgumentList $arguments `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr
[pscustomobject]@{
    process_id = $process.Id
    stdout = $stdout
    stderr = $stderr
    output_root = $outputRoot
    arm = 'candidate'
} | ConvertTo-Json -Compress

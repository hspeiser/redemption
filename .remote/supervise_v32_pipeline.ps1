$ErrorActionPreference = 'Stop'

$repo = 'C:\Users\henry\aigp'
$python = Join-Path $repo '.venv-train\Scripts\python.exe'
$out = Join-Path $repo 'worldmodel\v32_allgate_registry_flywheel'
$status = Join-Path $out 'pipeline_status.json'
New-Item -ItemType Directory -Force $out | Out-Null

function Write-PipelineStatus {
    param([string]$Stage, [string]$Detail = '', [int]$ExitCode = -1)
    $payload = [ordered]@{
        stage = $Stage
        detail = $Detail
        updated_utc = [DateTime]::UtcNow.ToString('o')
        supervisor_pid = $PID
    }
    if ($ExitCode -ge 0) {
        $payload.exit_code = $ExitCode
    }
    $temporary = "$status.tmp"
    $payload | ConvertTo-Json | Set-Content -LiteralPath $temporary
    Move-Item -Force -LiteralPath $temporary -Destination $status
}

function Active-ControlProofs {
    return @(
        Get-CimInstance Win32_Process | Where-Object {
            $_.CommandLine -match 'lrspeiser.*control_proof.py'
        }
    )
}

# Require a sustained idle window so we do not start in the small reset gap
# between another campaign's flights.
$idleChecks = 0
while ($idleChecks -lt 4) {
    $busy = @(Active-ControlProofs)
    if ($busy.Count) {
        $idleChecks = 0
        Write-PipelineStatus 'waiting_for_gipsy' (
            "control-proof processes active: " +
            (($busy | ForEach-Object ProcessId) -join ',')
        )
    } else {
        $idleChecks += 1
        Write-PipelineStatus 'idle_confirmation' "$idleChecks/4 checks"
    }
    Start-Sleep -Seconds 15
}

$trainArgs = @(
    '-u', 'scripts\train_vq2_g0g4_worldmodel.py',
    '--dataset', 'worldmodel\g0g16_master_currentera_v32_registry',
    '--audit-dataset', 'worldmodel\g0g16_allera_registry_audit_v2',
    '--base-model', 'data\fastsim_model_v2.json',
    '--map', 'data\vq2_runtime_map_g9g15fix.json',
    '--init-ensemble',
    'worldmodel\v30_allgate_registry_flywheel\residual_ensemble_v30.pt',
    '--out',
    'worldmodel\v32_allgate_registry_flywheel\residual_ensemble_v32.pt',
    '--members', '5', '--epochs', '25', '--batch-size', '2048',
    '--lr', '8e-5', '--rollout-finetune-epochs', '1',
    '--rollout-horizons', '8,16,32', '--rollout-lr', '2e-5',
    '--rollout-batch-size', '256', '--rollout-max-starts', '16384',
    '--seed', '20260832', '--device', 'cuda'
)
Write-PipelineStatus 'training' 'v32 five-member ensemble'
$train = Start-Process -FilePath $python -ArgumentList $trainArgs `
    -WorkingDirectory $repo -WindowStyle Hidden -Wait -PassThru `
    -RedirectStandardOutput (Join-Path $out 'stdout.log') `
    -RedirectStandardError (Join-Path $out 'stderr.log')
if ($train.ExitCode -ne 0) {
    Write-PipelineStatus 'training_failed' 'see stderr.log' $train.ExitCode
    exit $train.ExitCode
}

$auditArgs = @(
    '-u', 'scripts\compare_vq2_worldmodels.py',
    '--model', 'v25=worldmodel\v25_allgate_v8\residual_ensemble_v25.pt',
    '--model', 'v28=worldmodel\v28_allgate_registry_flywheel\residual_ensemble_v28.pt',
    '--model', 'v30=worldmodel\v30_allgate_registry_flywheel\residual_ensemble_v30.pt',
    '--model', 'v32=worldmodel\v32_allgate_registry_flywheel\residual_ensemble_v32.pt',
    '--dataset', 'worldmodel\g0g16_allera_registry_audit_v2',
    '--base-model', 'data\fastsim_model_v2.json',
    '--map', 'data\vq2_runtime_map_g9g15fix.json',
    '--out',
    'worldmodel\v32_allgate_registry_flywheel\fresh_registry_audit_h32.json',
    '--horizon', '32', '--device', 'cuda'
)
Write-PipelineStatus 'auditing' 'frozen v25/v28/v30/v32 comparison'
$audit = Start-Process -FilePath $python -ArgumentList $auditArgs `
    -WorkingDirectory $repo -WindowStyle Hidden -Wait -PassThru `
    -RedirectStandardOutput (Join-Path $out 'audit.stdout.log') `
    -RedirectStandardError (Join-Path $out 'audit.stderr.log')
if ($audit.ExitCode -ne 0) {
    Write-PipelineStatus 'audit_failed' 'see audit.stderr.log' $audit.ExitCode
    exit $audit.ExitCode
}

Write-PipelineStatus 'complete' 'v32 trained and frozen audit complete' 0

param(
    [string]$Candidate = 'D:\ai-gp\worldmodel\g0g4_v22only_tier87_cem_v1.json',
    [string]$Checkpoint = 'C:\Users\henry\Desktop\ai-gp\worldmodel\ppo_multimodel_segmentcredit_v8\best.pt',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\henry\Desktop\ai-gp'
$baselineLauncher = Join-Path $repo '.remote\launch_live_g0g4_baseline_control.ps1'
$candidateLauncher = Join-Path $repo '.remote\launch_live_g0g4_geometry_candidate.ps1'
$reportRoot = 'D:\ai-gp\training\vq2_g0g4_ab_v1'
$candidateRoot = 'D:\ai-gp\training\vq2_g0g4_geometry_live'
$baselineRoot = 'D:\ai-gp\training\vq2_g0g4_ab_v1_baseline'
$sequence = @(
    'baseline', 'candidate', 'candidate', 'baseline',
    'baseline', 'candidate', 'candidate', 'baseline'
)

foreach ($required in @($baselineLauncher, $candidateLauncher, $Candidate, $Checkpoint)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required file not found: $required"
    }
}

if ($DryRun) {
    [pscustomobject]@{
        sequence = $sequence
        candidate = (Resolve-Path -LiteralPath $Candidate).Path
        checkpoint = (Resolve-Path -LiteralPath $Checkpoint).Path
        strict_timing_abort = $true
        simulator_restart = $false
    }
    exit 0
}

New-Item -ItemType Directory -Force -Path $reportRoot | Out-Null
$started = Get-Date
$results = @()
$status = 'completed'

for ($index = 0; $index -lt $sequence.Count; $index++) {
    $arm = $sequence[$index]
    $armStarted = Get-Date
    if ($arm -eq 'baseline') {
        $launch = (& $baselineLauncher -Episodes 1 -OutputRoot $baselineRoot) |
            ConvertFrom-Json
        $root = $baselineRoot
    } else {
        $launch = (& $candidateLauncher -Candidate $Candidate -Episodes 1 `
            -Checkpoint $Checkpoint -Multigate) | ConvertFrom-Json
        $root = $candidateRoot
    }
    $process = Get-Process -Id ([int]$launch.process_id) -ErrorAction Stop
    $process.WaitForExit()
    $exitCode = $process.ExitCode
    if ($null -ne $exitCode -and $exitCode -ne 0) {
        $status = "${arm}_process_failed"
        $results += [pscustomobject]@{
            index = $index
            arm = $arm
            exit_code = $exitCode
            stdout = $launch.stdout
            stderr = $launch.stderr
        }
        break
    }

    $run = Get-ChildItem -LiteralPath $root -Directory |
        Where-Object { $_.LastWriteTime -ge $armStarted.AddSeconds(-2) } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $run -or -not (Test-Path -LiteralPath (Join-Path $run.FullName 'episodes.jsonl'))) {
        $status = "${arm}_summary_missing"
        break
    }
    $summary = Get-Content -LiteralPath (Join-Path $run.FullName 'episodes.jsonl') -Tail 1 |
        ConvertFrom-Json
    $results += [pscustomobject]@{
        index = $index
        arm = $arm
        run = $run.FullName
        timing_healthy = [bool]$summary.timing_healthy
        timing_health_reasons = @($summary.timing_health_reasons)
        poc_completed = [bool]$summary.poc_completed
        gate_reached = [int]$summary.gate_reached
        time_s = if ($summary.poc_completed) { [double]$summary.duration_s } else { $null }
        sim_step_p95_s = [double]$summary.sim_step_p95_s
        sim_step_max_s = [double]$summary.sim_step_max_s
        failure = $summary.failure
        stdout = $launch.stdout
        stderr = $launch.stderr
    }
    if (-not [bool]$summary.timing_healthy) {
        $status = 'infrastructure_abort'
        break
    }
    Start-Sleep -Milliseconds 750
}

$report = [ordered]@{
    status = $status
    started = $started.ToString('o')
    finished = (Get-Date).ToString('o')
    sequence = $sequence
    candidate = (Resolve-Path -LiteralPath $Candidate).Path
    checkpoint = (Resolve-Path -LiteralPath $Checkpoint).Path
    results = $results
}
$reportPath = Join-Path $reportRoot ("abba_" + $started.ToString('yyyyMMdd_HHmmss') + '.json')
$report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $reportPath
[pscustomobject]@{
    status = $status
    report = $reportPath
    completed_arms = $results.Count
}

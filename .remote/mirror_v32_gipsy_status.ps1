param(
    [string]$Remote = 'henry@gipsydanger',
    [string]$StatePath = 'D:\ai-gp\pipeline_state.json',
    [int]$IntervalSeconds = 15
)

$ErrorActionPreference = 'Continue'
$remoteStatus = 'C:\Users\henry\aigp\worldmodel\v32_allgate_registry_flywheel\pipeline_status.json'
$terminal = @('complete', 'training_failed', 'audit_failed')

while ($true) {
    $raw = & ssh.exe -o BatchMode=yes -o ConnectTimeout=5 $Remote `
        cmd.exe /d /c type $remoteStatus 2>$null | Out-String
    try {
        $remoteState = $raw | ConvertFrom-Json -ErrorAction Stop
    } catch {
        Start-Sleep -Seconds ([Math]::Max(5, $IntervalSeconds))
        continue
    }

    $stage = [string]$remoteState.stage
    $activeStage = switch ($stage) {
        'auditing' { 'audit' }
        'complete' { 'decide' }
        'audit_failed' { 'audit' }
        default { 'train' }
    }
    $message = switch ($stage) {
        'waiting_for_gipsy' {
            "v32 queued until Gipsydanger's control-proof campaign is idle. $($remoteState.detail)"
        }
        'idle_confirmation' {
            "v32 waiting for a sustained idle window. $($remoteState.detail)"
        }
        'training' { 'v32 world-model ensemble is training on Gipsydanger.' }
        'auditing' { 'v32 is running the frozen v25/v28/v30 comparison audit.' }
        'complete' { 'v32 training and audit completed; promotion decision is next.' }
        'training_failed' { 'v32 training failed; inspect the remote stderr log.' }
        'audit_failed' { 'v32 audit failed; inspect the remote audit stderr log.' }
        default { "Gipsydanger pipeline state: $stage" }
    }
    $payload = [ordered]@{
        active_stage = $activeStage
        message = $message
        remote_stage = $stage
        remote_updated_utc = $remoteState.updated_utc
        updated_unix_s = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    }
    $temporary = "$StatePath.tmp"
    $payload | ConvertTo-Json | Set-Content -LiteralPath $temporary
    Move-Item -Force -LiteralPath $temporary -Destination $StatePath
    if ($stage -in $terminal) {
        break
    }
    Start-Sleep -Seconds ([Math]::Max(5, $IntervalSeconds))
}

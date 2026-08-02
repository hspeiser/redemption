$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\henry\Desktop\ai-gp'
$mirror = Join-Path $repo '.remote\mirror_v32_gipsy_status.ps1'
$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match 'mirror_v32_gipsy_status.ps1' -and
    $_.ProcessId -ne $PID
}
if ($existing) {
    [pscustomobject]@{
        state = 'already_running'
        process_ids = @($existing.ProcessId)
    } | ConvertTo-Json -Compress
    exit 0
}
$process = Start-Process -FilePath powershell.exe -ArgumentList @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $mirror
) -WorkingDirectory $repo -WindowStyle Hidden -PassThru
[pscustomobject]@{
    state = 'started'
    process_id = $process.Id
} | ConvertTo-Json -Compress

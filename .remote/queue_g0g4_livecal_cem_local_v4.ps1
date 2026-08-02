$ErrorActionPreference = 'Stop'
$mutex = [Threading.Mutex]::new($false, 'Global\AIGP_G0G4_LiveCal_CEM_Local_V4')
if (-not $mutex.WaitOne(0)) { exit 0 }
try {
$repo = 'C:\Users\henry\Desktop\ai-gp'
$previous = 'D:\ai-gp\worldmodel\g0g4_geometry_rate_actoraware_tier87_recovered_local_v3.json'
$logRoot = 'D:\ai-gp\worldmodel\g0g4_livecal_actoraware_tier87_cem_local_v4_logs'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
while (-not (Test-Path -LiteralPath $previous)) { Start-Sleep -Seconds 20 }
while (Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match 'optimize_vq2_g0g4_worldmodel.py' -and
    $_.CommandLine -match 'g0g4_geometry_rate_actoraware_tier87_recovered_local_v3.json'
}) { Start-Sleep -Seconds 20 }
$process = Start-Process -FilePath 'cmd.exe' `
    -ArgumentList '/d', '/c', (Join-Path $repo '.remote\run_g0g4_livecal_cem_local_v4.cmd') `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru -Wait `
    -RedirectStandardOutput (Join-Path $logRoot 'stdout.log') `
    -RedirectStandardError (Join-Path $logRoot 'stderr.log')
$exitCode = $process.ExitCode
} finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
exit $exitCode

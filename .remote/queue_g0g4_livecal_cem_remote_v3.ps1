$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\henry\aigp'
$previous = Join-Path $repo 'worldmodel\g0g4_geometry_rate_actoraware_tier87_recovered_remote_v2.json'
$logRoot = Join-Path $repo 'worldmodel\g0g4_livecal_actoraware_tier87_cem_remote_v3_logs'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
while (-not (Test-Path -LiteralPath $previous)) { Start-Sleep -Seconds 30 }
while (Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match 'optimize_vq2_g0g4_worldmodel.py' -and
    $_.CommandLine -match 'g0g4_geometry_rate_actoraware_tier87_recovered_remote_v2.json'
}) { Start-Sleep -Seconds 30 }
$process = Start-Process -FilePath 'cmd.exe' `
    -ArgumentList '/d', '/c', (Join-Path $repo '.remote\run_g0g4_livecal_cem_remote_v3.cmd') `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru -Wait `
    -RedirectStandardOutput (Join-Path $logRoot 'stdout.log') `
    -RedirectStandardError (Join-Path $logRoot 'stderr.log')
exit $process.ExitCode

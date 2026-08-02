$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\henry\aigp'
$out = Join-Path $repo 'worldmodel\g0g4_geometry_rate_actoraware_tier87_recovered_remote_v2.json'
$logRoot = Join-Path $repo 'worldmodel\g0g4_geometry_rate_actoraware_tier87_recovery_remote_logs'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
while (Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match 'optimize_vq2_g0g4_worldmodel.py' -and
    $_.CommandLine -match 'g0g4_geometry_rate_actoraware_tier87_cem_v2.json'
}) { Start-Sleep -Seconds 30 }
if (Test-Path -LiteralPath $out) { exit 0 }
$process = Start-Process -FilePath 'cmd.exe' `
    -ArgumentList '/d', '/c', (Join-Path $repo '.remote\run_g0g4_recover_remote_v2.cmd') `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru -Wait `
    -RedirectStandardOutput (Join-Path $logRoot 'stdout.log') `
    -RedirectStandardError (Join-Path $logRoot 'stderr.log')
exit $process.ExitCode

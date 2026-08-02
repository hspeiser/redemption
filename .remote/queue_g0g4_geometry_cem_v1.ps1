$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\henry\aigp'
$logRoot = Join-Path $repo 'worldmodel\g0g4_geometry_actoraware_tier87_cem_v1_logs'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

while (Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match 'fastsim_train_ppo.py' -and
    $_.CommandLine -match 'ppo_all17_multimodel_v3_late_safe'
}) {
    Start-Sleep -Seconds 30
}

$process = Start-Process -FilePath 'cmd.exe' `
    -ArgumentList '/d', '/c', (Join-Path $repo '.remote\run_g0g4_geometry_cem_v1.cmd') `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru -Wait `
    -RedirectStandardOutput (Join-Path $logRoot 'stdout.log') `
    -RedirectStandardError (Join-Path $logRoot 'stderr.log')
exit $process.ExitCode

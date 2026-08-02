$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\henry\aigp'
$out = Join-Path $repo 'worldmodel\v23_candidate_live\residual_ensemble_v23.pt'
$logs = Join-Path $repo 'worldmodel\v23_candidate_live\logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
if (Test-Path -LiteralPath $out) { exit 0 }
$process = Start-Process -FilePath 'cmd.exe' `
    -ArgumentList '/d', '/c', (Join-Path $repo '.remote\run_train_v23_candidate_live.cmd') `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru -Wait `
    -RedirectStandardOutput (Join-Path $logs 'stdout.log') `
    -RedirectStandardError (Join-Path $logs 'stderr.log')
exit $process.ExitCode

$ErrorActionPreference = 'Stop'
$root = 'C:\Users\henry\aigp_dyna_r1'
$python = 'C:\Users\henry\aigp\.venv-train\Scripts\python.exe'
$outDir = Join-Path $root 'worldmodel\v25_live_abba_v7'
$stdout = Join-Path $outDir 'stdout.log'
$stderr = Join-Path $outDir 'stderr.log'
$runner = Join-Path $root 'run_dyna_r1_gipsy.ps1'
$arguments = @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $runner
)
$process = Start-Process -FilePath 'powershell.exe' -ArgumentList $arguments `
    -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr
[pscustomobject]@{
    process_id = $process.Id
    stdout = $stdout
    stderr = $stderr
    output = Join-Path $outDir 'residual_ensemble_v25.pt'
} | ConvertTo-Json -Compress

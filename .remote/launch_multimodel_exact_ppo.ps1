$ErrorActionPreference = "Stop"

$repo = "C:\Users\henry\aigp"
$runDir = Join-Path $repo "worldmodel\ppo_multimodel_cemsafe_v2"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$process = Start-Process `
    -FilePath "$env:SystemRoot\System32\cmd.exe" `
    -ArgumentList "/c", (Join-Path $repo "run_multimodel_exact_ppo.cmd") `
    -WorkingDirectory $repo `
    -WindowStyle Hidden `
    -PassThru
$process.Id | Set-Content (Join-Path $runDir "launcher_pid.txt")
Write-Output $process.Id

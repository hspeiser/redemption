$root = 'C:\Users\henry\aigp_dyna_r1\worldmodel\v25_live_abba_v7'
$processes = Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -eq 'python.exe' -or
        $_.CommandLine -like '*train_vq2_g0g4_worldmodel.py*' -or
        $_.CommandLine -like '*aigp_dyna_r1*'
    } |
    Select-Object ProcessId, Name, CommandLine
$stdout = Join-Path $root 'stdout.log'
$stderr = Join-Path $root 'stderr.log'
[pscustomobject]@{
    processes = @($processes)
    stdout_tail = if (Test-Path $stdout) { @(Get-Content $stdout -Tail 25) } else { @() }
    stderr_tail = if (Test-Path $stderr) { @(Get-Content $stderr -Tail 25) } else { @() }
    output_exists = Test-Path (Join-Path $root 'residual_ensemble_v25.pt')
} | ConvertTo-Json -Depth 5

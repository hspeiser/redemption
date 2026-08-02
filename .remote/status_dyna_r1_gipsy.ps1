$items = Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like '*train_vq2_g0g4_worldmodel.py*' }
$rows = foreach ($item in $items) {
    $process = Get-Process -Id $item.ProcessId -ErrorAction SilentlyContinue
    [pscustomobject]@{
        pid = $item.ProcessId
        cpu_s = if ($process) { $process.CPU } else { $null }
        working_set_mb = if ($process) {
            [Math]::Round($process.WorkingSet64 / 1MB, 1)
        } else { $null }
    }
}
[pscustomobject]@{
    processes = @($rows)
    partial = Test-Path 'C:\Users\henry\aigp_dyna_r1\worldmodel\v25_live_abba_v7\residual_ensemble_v25.partial.pt'
    final = Test-Path 'C:\Users\henry\aigp_dyna_r1\worldmodel\v25_live_abba_v7\residual_ensemble_v25.pt'
} | ConvertTo-Json -Depth 4

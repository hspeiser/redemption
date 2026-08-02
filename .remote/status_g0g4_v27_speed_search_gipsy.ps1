Get-Process -Name python -ErrorAction SilentlyContinue |
    Select-Object Id, CPU, WorkingSet64, StartTime, Responding |
    Format-Table -AutoSize
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Select-Object ProcessId, ParentProcessId, CreationDate, CommandLine |
    Format-List
Get-Item `
    'C:\Users\henry\aigp_dyna_r1\worldmodel\g0g4_v27_speed_search.stdout.log', `
    'C:\Users\henry\aigp_dyna_r1\worldmodel\g0g4_v27_speed_search.stderr.log' `
    -ErrorAction SilentlyContinue |
    Select-Object Name, Length, LastWriteTime |
    Format-Table -AutoSize
$checkpointPath = 'C:\Users\henry\aigp_dyna_r1\worldmodel\g0g4_v27_speed_search_v25v26_checkpoint.json'
if (Test-Path -LiteralPath $checkpointPath) {
    $candidate = Get-Content -LiteralPath $checkpointPath -Raw | ConvertFrom-Json
    Write-Output "iteration=$($candidate.source_iteration)"
    $candidate.champion_by_model | ForEach-Object {
        [pscustomobject]@{
            finish = $_.finish_rate
            tier = $_.tier_success_rate
            median = $_.median_s
            p90 = $_.p90_s
            clearance = $_.clearance_p10_m
        }
    } | Format-Table -AutoSize
}
Get-Item `
    'C:\Users\henry\aigp_dyna_r1\worldmodel\g0g4_v27_speed_search_v25v26.json', `
    'C:\Users\henry\aigp_dyna_r1\worldmodel\g0g4_v27_speed_search_v25v26_checkpoint.json' `
    -ErrorAction SilentlyContinue |
    Select-Object Name, Length, LastWriteTime |
    Format-Table -AutoSize

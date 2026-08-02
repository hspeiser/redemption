$ErrorActionPreference = 'SilentlyContinue'
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Select-Object ProcessId, ParentProcessId, CreationDate, CommandLine
Get-Process -Id 28744,33692 |
    Select-Object Id, CPU, WorkingSet64, StartTime, Responding |
    Format-List
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
$out = 'C:\Users\henry\aigp_dyna_r1\worldmodel\g0g4_dyna_r1_v24v25_cem_v1.json'
if (Test-Path -LiteralPath $out) {
    Get-Item -LiteralPath $out | Select-Object FullName, Length, LastWriteTime
}

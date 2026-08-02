param(
    [string]$HostAddress = '127.0.0.1',
    [int]$Port = 8900
)

$existing = Get-NetTCPConnection -LocalPort $Port -State Listen `
    -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -ne $existing) {
    [pscustomobject]@{
        process_id = $existing.OwningProcess
        url = "http://${HostAddress}:$Port/"
        already_running = $true
    } | ConvertTo-Json -Compress
    exit 0
}

$repo = 'C:\Users\henry\Desktop\ai-gp'
$python = Join-Path $repo '.venv-train\Scripts\python.exe'
$script = Join-Path $repo 'scripts\serve_vq2_pipeline_dashboard.py'
$logRoot = 'D:\ai-gp\runlogs'
New-Item -ItemType Directory -Force $logRoot | Out-Null
$process = Start-Process -FilePath $python `
    -ArgumentList @('-u', $script, '--host', $HostAddress, '--port', $Port) `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logRoot 'vq2_pipeline_dashboard.stdout.log') `
    -RedirectStandardError (Join-Path $logRoot 'vq2_pipeline_dashboard.stderr.log')
[pscustomobject]@{
    process_id = $process.Id
    url = "http://${HostAddress}:$Port/"
    already_running = $false
} | ConvertTo-Json -Compress

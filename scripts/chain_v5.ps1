# Waits for the v4 fine-tune to finish, preserves its checkpoints under
# versioned names, then launches the from-scratch v5 run detached.
$repo = "C:\Users\henry\Desktop\ai-gp"
$v4log = Join-Path $repo "data\models\train_v4.log"
$marker = Join-Path $repo "data\models\chain_v5.log"

"chainer started $(Get-Date -Format o)" | Out-File $marker -Encoding utf8

while ($true) {
    if ((Test-Path $v4log) -and (Select-String -Path $v4log -Pattern "TRAINING DONE" -Quiet)) {
        break
    }
    Start-Sleep -Seconds 20
}

"v4 done $(Get-Date -Format o)" | Add-Content $marker
# preserve v4 checkpoints under versioned names
Copy-Item (Join-Path $repo "data\models\gatenet_best.pt") (Join-Path $repo "data\models\gatenet_v4_best.pt") -Force
Copy-Item (Join-Path $repo "data\models\gatenet_last.pt") (Join-Path $repo "data\models\gatenet_v4_last.pt") -Force
"v4 checkpoints preserved" | Add-Content $marker

Start-Process -FilePath (Join-Path $repo ".venv-train\Scripts\python.exe") `
    -ArgumentList "scripts\train_net.py","--epochs","45","--batch","16","--workers","6","--lr","3e-4","--tag","v5" `
    -WorkingDirectory $repo `
    -RedirectStandardOutput (Join-Path $repo "data\models\train_v5.log") `
    -RedirectStandardError (Join-Path $repo "data\models\train_v5.err") `
    -WindowStyle Hidden
"v5 launched $(Get-Date -Format o)" | Add-Content $marker

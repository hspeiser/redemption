# Poll gipsydanger training + spawn-start eval until done/milestone.
$log = "C:\Users\henry\Desktop\ai-gp\.remote\monitor_ppo.log"
"monitor start $(Get-Date)" | Out-File $log -Encoding utf8
for ($i = 0; $i -lt 12; $i++) {
    Start-Sleep -Seconds 900
    $out = ssh henry@gipsydanger "wsl bash /mnt/c/Users/henry/check_ppo.sh" 2>&1 | Out-String
    "===== check $i $(Get-Date) =====`n$out" | Out-File $log -Append -Encoding utf8
    if ($out -notmatch "fastsim_train_ppo") {
        "TRAINER NOT RUNNING - stopping monitor" | Out-File $log -Append -Encoding utf8
        break
    }
    if ($out -match "FINISHED FULL COURSE: (\d+)/1024") {
        $n = [int]$Matches[1]
        if ($n -ge 920) {
            "MILESTONE: $n/1024 finished - stopping monitor" | Out-File $log -Append -Encoding utf8
            break
        }
    }
}
"monitor end $(Get-Date)" | Out-File $log -Append -Encoding utf8

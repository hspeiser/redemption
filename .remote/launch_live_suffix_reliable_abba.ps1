param(
    [int]$Cycles = 2,
    [switch]$DryRun
)

$repo = 'C:\Users\henry\Desktop\ai-gp'
$launcher = Join-Path $repo '.remote\launch_live_full17_fastprefix_abba.ps1'
$candidate = Join-Path $repo 'worldmodel\suffix_exact_straights_v1\speed_config.json'
$lateral = '0:0,1:0.300000012,2:0.0878505111,3:-0.3,4:0.45,5:0.1,6:-0.1,11:0.15404789,12:0.155685414,13:-0.0702574872,14:-0.262386669,15:0.15,16:0.100440218'
$velocity = '0:1.53180218,1:1.82969701,2:1.49190915,3:1.31209302,4:2.20000005,11:1.014624055,12:1.031248149,13:1.05,14:1,15:1.062275906,16:1.198293646'
$leads = '0:1,1:4,2:14,3:1,4:2,11:2,12:1,13:0,14:0,15:0,16:2'

& $launcher -Cycles $Cycles -CandidateConfig $candidate `
    -ReferenceLateralOffsets $lateral `
    -ReferenceVelocityScales $velocity `
    -ReferenceActionLeads $leads -DryRun:$DryRun

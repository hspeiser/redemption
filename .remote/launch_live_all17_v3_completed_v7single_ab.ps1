param(
    [int]$Episodes = 2,
    [string]$Checkpoint = 'D:\ai-gp\worldmodel\ppo_all17_multimodel_v3_late_safe\best_iter250.pt',
    [string]$ResidualGates = '6,7,8,9,10,11,12,13,14',
    [string]$Primary = 'C:\Users\henry\Desktop\ai-gp\data\models\gatenet_v7_best.pt',
    [string]$FrozenActionEpisode = '',
    [string]$FrozenActionGates = '',
    [switch]$Multigate,
    [switch]$CandidateOnly
)

$ErrorActionPreference = 'Stop'

$repo = 'C:\Users\henry\Desktop\ai-gp'
$python = Join-Path $repo '.venv-train\Scripts\python.exe'
$launcher = Join-Path $repo 'scripts\launch_vq2_from_config.py'
$config = 'D:\ai-gp\training\vq2_teachercheck2\20260731_151519\config.json'
$outputRoot = 'D:\ai-gp\training\vq2_all17_ppo_v3_completed_v7single_live_ab'
$map = Join-Path $repo 'data\vq2_runtime_map_g9g15fix.json'
$logRoot = 'D:\ai-gp\runlogs'
$interleave = if ($CandidateOnly) { 'false' } else { 'true' }

New-Item -ItemType Directory -Force -Path $outputRoot, $logRoot | Out-Null
$env:AIGP_MULTIGATE = if ($Multigate) { '1' } else { '0' }

$arguments = @(
    '-u', $launcher,
    '--config', $config,
    '--output-root', $outputRoot,
    '--episodes', "$Episodes",
    '--eval-only',
    '--override', "ppo_residual_checkpoint=$Checkpoint",
    '--override', "residual_gates=$ResidualGates",
    '--override', 'train_gate=-1',
    '--override', 'residual_scale=0.20',
    '--override', "interleave_protected_champion=$interleave",
    '--override', "map=$map",
    '--override', "primary=$Primary",
    '--override', 'right_lateral_action_bias=0.0'
)

if ($FrozenActionEpisode) {
    $arguments += @(
        '--override', "frozen_action_episode=$FrozenActionEpisode",
        '--override', "frozen_action_gates=$FrozenActionGates"
    )
}

Start-Process -FilePath $python -ArgumentList $arguments `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logRoot 'vq2_all17_ppo_v3_completed_v7single_ab.stdout.log') `
    -RedirectStandardError (Join-Path $logRoot 'vq2_all17_ppo_v3_completed_v7single_ab.stderr.log') | Select-Object -ExpandProperty Id

$ErrorActionPreference = 'Stop'

$repo = 'C:\Users\henry\Desktop\ai-gp'
$python = Join-Path $repo '.venv-train\Scripts\python.exe'
$launcher = Join-Path $repo 'scripts\launch_vq2_from_config.py'
$config = 'D:\ai-gp\training\vq2_fullcourse_reliablebackbone_all17_v1\20260801_160352\config.json'
$outputRoot = 'D:\ai-gp\training\vq2_all17_ppo_v1_live_ab'
$checkpoint = 'D:\ai-gp\worldmodel\ppo_all17_multimodel_v1\best_iter120.pt'
$primary = Join-Path $repo 'data\models\gatenet_v13drought_best.pt'
$logRoot = 'D:\ai-gp\runlogs'

New-Item -ItemType Directory -Force -Path $outputRoot, $logRoot | Out-Null
$env:AIGP_MULTIGATE = '1'

$arguments = @(
    '-u', $launcher,
    '--config', $config,
    '--output-root', $outputRoot,
    '--episodes', '8',
    '--eval-only',
    '--override', "ppo_residual_checkpoint=$checkpoint",
    '--override', 'residual_gates=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16',
    '--override', 'train_gate=-1',
    '--override', 'residual_scale=0.20',
    '--override', 'interleave_protected_champion=true',
    '--override', "primary=$primary"
)

$process = Start-Process -FilePath $python -ArgumentList $arguments `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logRoot 'vq2_all17_ppo_v1_live_ab.stdout.log') `
    -RedirectStandardError (Join-Path $logRoot 'vq2_all17_ppo_v1_live_ab.stderr.log')
$process.Id

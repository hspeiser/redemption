$ErrorActionPreference = 'Stop'

$repo = 'C:\Users\henry\Desktop\ai-gp'
$python = Join-Path $repo '.venv-train\Scripts\python.exe'
$launcher = Join-Path $repo 'scripts\launch_vq2_from_config.py'
$config = 'D:\ai-gp\training\vq2_teachercheck2\20260731_151519\config.json'
$outputRoot = 'D:\ai-gp\training\vq2_all17_ppo_v3_completed_backbone_live_ab'
$checkpoint = 'D:\ai-gp\worldmodel\ppo_all17_multimodel_v3_late_safe\best_iter250.pt'
$map = Join-Path $repo 'data\vq2_runtime_map_g9g15fix.json'
$primary = Join-Path $repo 'data\models\gatenet_v13drought_best.pt'
$logRoot = 'D:\ai-gp\runlogs'

New-Item -ItemType Directory -Force -Path $outputRoot, $logRoot | Out-Null
$env:AIGP_MULTIGATE = '1'

$arguments = @(
    '-u', $launcher,
    '--config', $config,
    '--output-root', $outputRoot,
    '--episodes', '12',
    '--eval-only',
    '--override', "ppo_residual_checkpoint=$checkpoint",
    '--override', 'residual_gates=6,7,8,9,10,11,12,13,14,15,16',
    '--override', 'train_gate=-1',
    '--override', 'residual_scale=0.20',
    '--override', 'interleave_protected_champion=true',
    '--override', "map=$map",
    '--override', "primary=$primary",
    '--override', 'right_lateral_action_bias=0.0'
)

Start-Process -FilePath $python -ArgumentList $arguments `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logRoot 'vq2_all17_ppo_v3_completed_backbone_ab.stdout.log') `
    -RedirectStandardError (Join-Path $logRoot 'vq2_all17_ppo_v3_completed_backbone_ab.stderr.log') | Select-Object -ExpandProperty Id

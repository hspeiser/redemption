$root = 'C:\Users\henry\aigp_dyna_r1'
$python = 'C:\Users\henry\aigp\.venv-train\Scripts\python.exe'
$stdout = Join-Path $root 'worldmodel\g0g4_v27_speed_search.stdout.log'
$stderr = Join-Path $root 'worldmodel\g0g4_v27_speed_search.stderr.log'
$arguments = @(
    '-u', 'scripts\optimize_vq2_g0g4_worldmodel.py',
    '--dataset', 'worldmodel\g0g4_current_aug_21_live_abba_v7_longroll',
    '--ensemble', 'worldmodel\v25_live_abba_v7\residual_ensemble_v25.pt',
    '--ensemble', 'worldmodel\v26_live_abba_v7_longroll\residual_ensemble_v26.pt',
    '--model', 'data\fastsim_model_v2.json',
    '--controller-model', 'data\fastsim_model_v2.json',
    '--demo', 'data\vq2_g0g1fast9267_g2plus_clean_demo.npz',
    '--map', 'data\vq2_runtime_map_g9g15fix.json',
    '--obstacles', 'data\vq2_obstacles_inflated.json',
    '--teacher-config', 'worldmodel\teacher_full17_cem_safe_v1.json',
    '--initial', 'worldmodel\g0g4_v22_g3center_schedule_v1.json',
    '--actor', 'worldmodel\ppo_multimodel_segmentcredit_v8\best.pt',
    '--residual-scale', '0.20',
    '--active-gates', '3', '4',
    '--geometry-gates', '3', '4',
    '--geometry-limit', '0.30',
    '--multigate-vision',
    '--population', '96', '--elite', '16', '--iterations', '24',
    '--worlds', '128', '--selection-worlds', '2048',
    '--final-worlds', '4096', '--clean-worlds', '1024',
    '--aleatoric-scale', '1.5', '--impulse-rate-hz', '0.08',
    '--live-estimator-realism', '--reliability-floor', '0.95',
    '--clearance-floor', '0.18', '--time-weight', '80',
    '--tier-target', '8.7', '--tier-bonus', '400',
    '--seed', '20261002', '--device', 'cuda',
    '--out', 'worldmodel\g0g4_v27_speed_search_v25v26.json'
)
$process = Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru
Write-Output $process.Id

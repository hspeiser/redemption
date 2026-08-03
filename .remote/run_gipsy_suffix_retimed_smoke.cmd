@echo off
cd /d C:\Users\henry\aigp
.venv-train\Scripts\python.exe scripts\fastsim_suffix_covis.py ^
  --mode speed --exact-search-space line --freeze-offsets ^
  --speed-gates 11,12,15,16 --line-a-fwd 5.0 --line-a-brk 6.0 ^
  --finalists 4 --parallel-evals 4 --reliability-floor 0.70 ^
  --model data\fastsim_model_v3_live.json ^
  --live-teacher-config worldmodel\gipsy_geometry_config.json ^
  --ensemble worldmodel\v25_allgate_v8\residual_ensemble_v25.pt worldmodel\v28_allgate_registry_flywheel\residual_ensemble_v28.pt worldmodel\v30_allgate_registry_flywheel\residual_ensemble_v30.pt ^
  --prefix-demo data\vq2_hybrid_fastprefix_a2suffix_g11_v2.npz ^
  --handoff-pool data\lineopt\handoff_pool_13finish.json ^
  --device cuda --seed 20261320 ^
  --out-prefix worldmodel\suffix_exact_retimed_smoke\speed --smoke

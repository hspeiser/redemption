@echo off
cd /d "C:\Users\satas\Downloads\AI-GP Simulator v1.0.3385-VQ1\PyAIPilotExample-v1"
set PYTHONIOENCODING=utf-8
"C:\Users\satas\projects\ai-grand-prix\.venv\Scripts\python.exe" -u -m racer_mujoco.train_mj --envs 64 --learner 1 --utd_cap 2.0 --min_utd 0.06 --cdrop 0 --sil_every 2 --batch 1024 --ent -2 --away 15 --nstep 5 --her 2 --silw 1.0 --gates 6 --episode_s 40 --mirror 1 --video_every 1000 > "C:\Users\satas\Downloads\AI-GP Simulator v1.0.3385-VQ1\PyAIPilotExample-v1\racer_mujoco\runs_mj\full.log" 2>&1

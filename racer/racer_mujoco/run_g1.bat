@echo off
cd /d "C:\Users\satas\Downloads\AI-GP Simulator v1.0.3385-VQ1\PyAIPilotExample-v1"
set PYTHONIOENCODING=utf-8
"C:\Users\satas\projects\ai-grand-prix\.venv\Scripts\python.exe" -u -m racer_mujoco.train_mj --envs 64 --updates 3 --batch 1024 --ent -2 --away 15 --nstep 5 --her 1 --silw 1.0 --gates 2 --episode_s 20 --resume racer_mujoco\runs_mj\mj_gate0_stackC.pt --cwarm 2000 --alpha0 0.05 --video_every 1000 > "C:\Users\satas\Downloads\AI-GP Simulator v1.0.3385-VQ1\PyAIPilotExample-v1\racer_mujoco\runs_mj\g1.log" 2>&1

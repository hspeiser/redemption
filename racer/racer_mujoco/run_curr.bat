@echo off
cd /d "C:\Users\satas\Downloads\AI-GP Simulator v1.0.3385-VQ1\PyAIPilotExample-v1"
set PYTHONIOENCODING=utf-8
"C:\Users\satas\projects\ai-grand-prix\.venv\Scripts\python.exe" -u -m racer_mujoco.train_mj --envs 64 --updates 8 --curr --curr_start 4 --curr_step 2.5 --away 15 --ent -2 > "C:\Users\satas\Downloads\AI-GP Simulator v1.0.3385-VQ1\PyAIPilotExample-v1\racer_mujoco\runs_mj\curr.log" 2>&1

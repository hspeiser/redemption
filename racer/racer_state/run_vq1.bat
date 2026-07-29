@echo off
cd /d "C:\Users\satas\Downloads\AI-GP Simulator v1.0.3385-VQ1\PyAIPilotExample-v1"
set PYTHONIOENCODING=utf-8
"C:\Users\satas\projects\ai-grand-prix\.venv\Scripts\python.exe" -u -m racer_state.train2 --resume "C:\Users\satas\Downloads\AI-GP Simulator v1.0.3385-VQ1\PyAIPilotExample-v1\racer_state\runs\autosave_vq1.pt" --cwarm 500 --her 2 --mirror 1 --silw 1.0 --utd_cap 1.0 --gates 6 --episode_s 30 --eval_every 25 --hz 30 > "C:\Users\satas\Downloads\AI-GP Simulator v1.0.3385-VQ1\PyAIPilotExample-v1\racer_state\runs\train2.log" 2>&1

$ErrorActionPreference = "Stop"
$dir = "C:\Users\satas\Downloads\AI-GP Simulator v1.0.3385-VQ1\PyAIPilotExample-v1"
$py  = "C:\Users\satas\projects\ai-grand-prix\.venv\Scripts\python.exe"
$log = "$dir\racer_mujoco\runs_mj\curr.log"
$env:PYTHONIOENCODING = "utf-8"
$args = "-u -m racer_mujoco.train_mj --envs 64 --updates 8 --curr --curr_start 4 --curr_step 2.5 --away 15 --ent -4 --alpha0 0.1"
$cmd  = "cmd.exe /c `"cd /d `"`"$dir`"`" && set PYTHONIOENCODING=utf-8 && `"`"$py`"`" $args > `"`"$log`"`" 2>&1`""
$p = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd }
Write-Output "started pid=$($p.ProcessId) rc=$($p.ReturnValue) log=$log"
